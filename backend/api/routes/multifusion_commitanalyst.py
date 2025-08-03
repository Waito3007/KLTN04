from services.repo_service import fetch_repo_from_github
import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional
import logging
from collections import defaultdict
from db.database import get_db
from services.multifusion_commitanalyst_service import MultifusionCommitAnalystService, MultiFusionV2Service
router = APIRouter(tags=["multifusion-commit-analysis"])
logger = logging.getLogger(__name__)
multifusion_v2 = MultiFusionV2Service()  # Ensure singleton instance is shared

@router.get("/{repo_id}/members/commits-analyst")
async def get_all_members_commits_analyst(
    repo_id: int,
    branch_name: str = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Lấy commits của tất cả member với MultiFusion V2 AI analysis"""
    try:
        service = MultifusionCommitAnalystService(db)
        # Lấy danh sách member từ bảng collaborators
        members_query = """
            SELECT github_username FROM collaborators WHERE repo_id = :repo_id
        """
        members = db.execute(text(members_query), {"repo_id": repo_id}).fetchall()
        member_logins = [m[0] for m in members if m[0]]
        results = []
        for member_login in member_logins:
            query = """
                SELECT id, sha, message, author_name, committer_date, branch_name, insertions, deletions, files_changed, diff_content
                FROM commits
                WHERE repo_id = :repo_id AND LOWER(author_name) = LOWER(:member_login)
            """
            params = {"repo_id": repo_id, "member_login": member_login, "limit": limit}
            if branch_name:
                query += " AND LOWER(branch_name) = LOWER(:branch_name)"
                params["branch_name"] = branch_name
            query += " ORDER BY committer_date DESC LIMIT :limit"
            commits = service._get_commits_from_db(query, params)
            if not commits:
                member_result = {
                    "member_login": member_login,
                    "total_commits": 0,
                    "commits": [],
                    "statistics": {}
                }
            else:
                commits_for_ai = service._format_commits_for_ai(commits)
                ai_results = service.multifusion_v2_service.predict_commit_type_batch(commits_for_ai)
                commits_with_analysis = service._combine_results(commits, ai_results)
                commit_type_stats = defaultdict(int)
                for c in commits_with_analysis:
                    commit_type_stats[c['analysis']['type']] += 1
                member_result = {
                    "member_login": member_login,
                    "total_commits": len(commits_with_analysis),
                    "commits": commits_with_analysis,
                    "statistics": {"commit_types": dict(commit_type_stats)}
                }
            results.append(member_result)
        return {"success": True, "data": results}
    except Exception as e:
        logger.error(f"Error analyzing all members commits with MultiFusion V2: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing all members commits with MultiFusion V2: {str(e)}")



@router.get("/{repo_id}/members/{member_login}/commits-analyst")
async def get_member_commits_analyst(
    repo_id: int,
    member_login: str,
    branch_name: str = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Lấy commits của member với MultiFusion V2 AI analysis"""
    try:
        service = MultifusionCommitAnalystService(db)
        # Lấy commit của member, lọc theo branch nếu có
        query = """
            SELECT id, sha, message, author_name, committer_date, branch_name, insertions, deletions, files_changed, diff_content
            FROM commits
            WHERE repo_id = :repo_id AND LOWER(author_name) = LOWER(:member_login)
        """
        params = {"repo_id": repo_id, "member_login": member_login, "limit": limit}
        if branch_name:
            query += " AND LOWER(branch_name) = LOWER(:branch_name)"
            params["branch_name"] = branch_name
        query += " ORDER BY committer_date DESC LIMIT :limit"
        commits = service._get_commits_from_db(query, params)
        if not commits:
            return {"success": True, "data": {"total_commits": 0, "commits": [], "statistics": {}}}
        commits_for_ai = service._format_commits_for_ai(commits)
        ai_results = service.multifusion_v2_service.predict_commit_type_batch(commits_for_ai)
        commits_with_analysis = service._combine_results(commits, ai_results)
        commit_type_stats = defaultdict(int)
        for c in commits_with_analysis:
            commit_type_stats[c['analysis']['type']] += 1
        return {
            "success": True,
            "data": {
                "total_commits": len(commits_with_analysis),
                "commits": commits_with_analysis,
                "statistics": {"commit_types": dict(commit_type_stats)}
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing member commits with MultiFusion V2: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing member commits with MultiFusion V2: {str(e)}")
# Direct diagnosis using GitHub API data

# Phân tích một commit 
@router.post("/{repo_id}/analyze-commit")
async def analyze_single_commit(repo_id: int, commit_data: dict):
    """
    Phân tích một commit sử dụng MultiFusion V2
    Đầu vào: message, file_count, lines_added, lines_removed, total_changes, num_dirs_changed
    """
    try:
        if not multifusion_v2.is_model_available():
            logger.warning("MultiFusion V2 model is not available for single commit analysis.")
            raise HTTPException(status_code=503, detail="MultiFusion V2 model is not available.")
        message = commit_data.get("message", "")
        file_count = commit_data.get("file_count", commit_data.get("files_count", 0))
        lines_added = commit_data.get("lines_added", 0)
        lines_removed = commit_data.get("lines_removed", 0)
        total_changes = commit_data.get("total_changes", lines_added + lines_removed)
        num_dirs_changed = commit_data.get("num_dirs_changed", 1)
        
        # Prepare single commit data as a list for batch prediction
        single_commit_for_prediction = [{
            "message": message,
            "file_count": file_count,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "total_changes": total_changes,
            "num_dirs_changed": num_dirs_changed
        }]
        
        logger.info(f"Input for single commit prediction: {single_commit_for_prediction}")
        results = multifusion_v2.predict_commit_type_batch(single_commit_for_prediction)
        result = results[0] if results else None # Extract the single result
        logger.info(f"Result from single commit prediction: {result}")
        
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Error analyzing single commit: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
# Batch phân tích nhiều commits 
@router.post("/{repo_id}/batch-analyze-commits")
async def batch_analyze_commits(repo_id: int, request_data: dict):
    """
    Batch phân tích nhiều commits sử dụng MultiFusion 
    Đầu vào: request_data["commits"]: list các dict với các trường như trên
    """
    try:
        if not multifusion_v2.is_model_available():
            raise HTTPException(status_code=503, detail="MultiFusion model is not available.")
        commits = request_data.get("commits", [])
        results = []
        for commit in commits:
            message = commit.get("message", "")
            file_count = commit.get("file_count", commit.get("files_count", 0))
            lines_added = commit.get("lines_added", 0)
            lines_removed = commit.get("lines_removed", 0)
            total_changes = commit.get("total_changes", lines_added + lines_removed)
            num_dirs_changed = commit.get("num_dirs_changed", 1)
            result = multifusion_v2.predict_commit_type(
                message, file_count, lines_added, lines_removed, total_changes, num_dirs_changed
            )
            results.append({"input": commit, "prediction": result})
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"Error batch analyzing commits: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
# Lấy tất cả commits của repo
@router.get("/{repo_id}/members/{member_login}/commits")
async def get_member_commits(repo_id: int, member_login: str, db: Session = Depends(get_db)):
    """
    Lấy danh sách commits của một member trong repo
    """
    try:
        service = MultifusionCommitAnalystService(db)
        commits = service.get_member_commits(repo_id, member_login)
        return {"success": True, "repository_id": repo_id, "member_login": member_login, "commits": commits}
    except Exception as e:
        logger.error(f"Error getting member commits: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
# Lấy tất cả commits của repo
@router.get("/{repo_id}/commits/all")
async def get_all_commits(repo_id: int, db: Session = Depends(get_db)):
    """
    Lấy tất cả commits của repo
    """
    try:
        service = MultifusionCommitAnalystService(db)
        commits = service.get_all_repo_commits_raw(repo_id)
        return {"success": True, "repository_id": repo_id, "commits": commits}
    except Exception as e:
        logger.error(f"Error getting all commits: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
# Lấy danh sách branches của repo từ DB
@router.get("/{repo_id}/branches")
async def get_branches(repo_id: int, db: Session = Depends(get_db)):
    """
    Lấy danh sách branches của repo
    """
    try:
        service = MultifusionCommitAnalystService(db)
        branches = service.get_branches(repo_id)
        return {"success": True, "repository_id": repo_id, "branches": branches}
    except Exception as e:
        logger.error(f"Error getting branches: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{repo_id}/ai/model-v2-status")
async def get_multifusion_v2_status(repo_id: int):
    """Kiểm tra trạng thái và thông tin của MultiFusion V2 model."""
    try:
        if not hasattr(multifusion_v2, "get_model_info"):
            logger.error("MultiFusionV2Service missing get_model_info method")
            raise HTTPException(status_code=500, detail="MultiFusionV2Service missing get_model_info method")
        model_info = multifusion_v2.get_model_info()
        return {"success": True, "repository_id": repo_id, "model_info": model_info}
    except Exception as e:
        try:
            logger.error(f"Error in get_multifusion_v2_status: {e}", exc_info=True)
        except Exception:
            print(f"Error in get_multifusion_v2_status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{repo_id}/commits/all/analysis")
async def get_all_repo_commits_analysis(
    repo_id: int,
    limit: int = 1000,  # Tăng default limit để lấy tất cả commits
    offset: int = 0,
    branch_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Lấy tất cả commits của repo với AI analysis."""
    try:
        service = MultifusionCommitAnalystService(db)
        analysis = await service.get_all_repo_commits_with_analysis(
            repo_id, limit, offset, branch_name
        )
        return {
            "success": True,
            "repository_id": repo_id,
            **analysis
        }
    except Exception as e:
        logger.error(f"Error getting all repo commits analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting all repo commits analysis: {str(e)}")