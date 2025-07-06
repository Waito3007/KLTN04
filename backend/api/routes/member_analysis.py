from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from db.database import get_db
from services.member_analysis_service import MemberAnalysisService
from services.multifusion_ai_service import MultiFusionAIService
from services.multifusion_v2_service import MultiFusionV2Service
from api.deps import get_multifusion_ai_service

router = APIRouter(prefix="/api/repositories", tags=["member-analysis"])

@router.get("/{repo_id}/members")
async def get_repository_members(
    repo_id: int,
    db: Session = Depends(get_db)
):
    """Lấy danh sách members của repository"""
    try:
        service = MemberAnalysisService(db)
        members = service.get_repository_members(repo_id)
        
        return {
            "success": True,
            "repository_id": repo_id,
            "members": members,
            "total": len(members)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching members: {str(e)}")

@router.get("/{repo_id}/members/{member_login}/commits")
async def get_member_commits_analysis(
    repo_id: int,
    member_login: str,
    branch_name: str = None,  # Optional branch filter
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Lấy commits của member với AI analysis (luôn sử dụng AI - MultiFusion V2)"""
    try:
        service = MemberAnalysisService(db)
        # Always use AI-powered analysis (MultiFusion V2)
        analysis = await service.get_member_commits_with_multifusion_v2_analysis(
            repo_id, member_login, limit, branch_name
        )

        # Đảm bảo trả về đúng định dạng frontend mong muốn
        # Nếu analysis là dict và có statistics thì trả về trực tiếp
        if isinstance(analysis, dict) and "statistics" in analysis:
            return {
                "success": True,
                "statistics": analysis["statistics"],
                "commits": analysis.get("commits", []),
                "branch_filter": branch_name
            }
        # Nếu analysis là list hoặc không có statistics, trả về như cũ
        return {
            "success": True,
            "data": analysis,
            "branch_filter": branch_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing member commits: {str(e)}")

@router.get("/{repo_id}/members/{member_login}/commits-han")
async def get_member_commits_han_analysis(
    repo_id: int,
    member_login: str,
    branch_name: str = None,  # Optional branch filter
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Lấy commits của member với HAN AI analysis"""
    try:
        service = MemberAnalysisService(db)
        analysis = await service.get_member_commits_with_ai_analysis(
            repo_id, member_login, limit, branch_name
        )
        return {
            "success": True,
            "data": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing member commits with HAN: {str(e)}")


@router.get("/{repo_id}/ai-features")
async def get_ai_features_status(repo_id: int):
    """Lấy status của các tính năng AI available"""
    return {
        "success": True,
        "repository_id": repo_id,
        "features": {
            "commit_analysis": True,
            "member_insights": True,
            "productivity_tracking": True,
            "code_pattern_detection": True,
            "han_model_analysis": True
        },
        "ai_model": {
            "name": "HAN Commit Analyzer",
            "version": "1.0",
            "type": "Hierarchical Attention Network",
            "capabilities": [
                "Deep commit message understanding",
                "Semantic commit classification",
                "Developer behavior analysis", 
                "Technology area detection",
                "Impact and urgency assessment",
                "Code quality insights"
            ]
        },
        "endpoints": {
            "commit_analysis": f"/api/repositories/{repo_id}/members/{{member_login}}/commits?use_ai=true",
            "batch_analysis": f"/api/repositories/{repo_id}/ai/analyze-batch",
            "developer_insights": f"/api/repositories/{repo_id}/ai/developer-insights"
        }
    }

@router.post("/{repo_id}/ai/analyze-batch")
async def analyze_commits_batch(
    repo_id: int,
    commit_messages: List[str],
    db: Session = Depends(get_db)
):
    """Batch analysis cho nhiều commit messages"""
    try:
        service = MemberAnalysisService(db)
        results = await service.ai_service.analyze_commits_batch(commit_messages)
        
        return {
            "success": True,
            "repository_id": repo_id,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in batch analysis: {str(e)}")

@router.get("/{repo_id}/ai/developer-insights")
async def get_developer_insights(
    repo_id: int,
    db: Session = Depends(get_db)
):
    """Lấy insights về tất cả developers trong repo"""
    try:
        service = MemberAnalysisService(db)
        
        # Get all members
        members = service.get_repository_members(repo_id)
        
        # Get commits for each member and analyze
        developer_commits = {}
        for member in members[:5]:  # Limit to first 5 for demo
            member_login = member['github_username']
            commits_data = service._get_member_commits_raw(repo_id, member_login, 20)
            if commits_data:
                developer_commits[member_login] = [row[2] for row in commits_data]  # messages
        
        # Analyze patterns
        insights = await service.ai_service.analyze_developer_patterns(developer_commits)
        
        return {
            "success": True,
            "repository_id": repo_id,
            "insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting developer insights: {str(e)}")

@router.post("/{repo_id}/ai/model-status")
async def get_ai_model_status(repo_id: int):
    """Kiểm tra trạng thái AI model (HAN Commit Analyzer)"""
    try:
        # Kiểm tra model HAN đã tồn tại chưa
        import os
        from pathlib import Path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        han_model_path = Path(current_dir).parent.parent / "ai" / "models" / "han_github_model" / "best_model.pth"
        model_loaded = han_model_path.exists()
        return {
            "success": True,
            "repository_id": repo_id,
            "model_loaded": model_loaded,
            "model_info": {
                "type": "Hierarchical Attention Network (HAN)",
                "purpose": "Commit message analysis and classification",
                "features": [
                    "Semantic understanding",
                    "Commit type classification",
                    "Technology area detection",
                    "Multi-task classification"
                ]
            }
        }
    except Exception as e:
        return {
            "success": False,
            "repository_id": repo_id,
            "model_loaded": False,
            "error": str(e)
        }

@router.post("/analyze-multifusion-commit")
async def analyze_multifusion_commit(
    commit_data: dict,
    multifusion_service: MultiFusionAIService = Depends(get_multifusion_ai_service)
):
    """
    Phân loại một commit sử dụng mô hình MultiFusion.
    
    Args:
        commit_data: Dữ liệu commit bao gồm commit_message, lines_added, lines_removed, files_count, detected_language.
    
    Returns:
        {"commit_type": string}
    """
    try:
        commit_message = commit_data.get("commit_message")
        lines_added = commit_data.get("lines_added")
        lines_removed = commit_data.get("lines_removed")
        files_count = commit_data.get("files_count")
        detected_language = commit_data.get("detected_language")

        if not all([commit_message, lines_added is not None, lines_removed is not None, files_count is not None, detected_language]):
            raise HTTPException(status_code=400, detail="Missing required commit data fields.")

        predicted_type = multifusion_service.predict_commit_type(
            commit_message, lines_added, lines_removed, files_count, detected_language
        )
        return {"commit_type": predicted_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing commit with MultiFusion model: {str(e)}")

@router.get("/{repo_id}/branches")
async def get_repository_branches(
    repo_id: int,
    db: Session = Depends(get_db)
):
    """Lấy danh sách branches của repository"""
    try:
        service = MemberAnalysisService(db)
        branches = service.get_repository_branches(repo_id)
        
        return {
            "success": True,
            "repository_id": repo_id,
            "branches": branches,
            "total": len(branches)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching branches: {str(e)}")

@router.post("/analyze-multifusion-v2-commit")
async def analyze_multifusion_v2_commit(commit_data: dict):
    """
    Phân loại một commit sử dụng mô hình MultiFusion V2 (BERT + MLP Fusion).
    
    Args:
        commit_data: {
            "commit_message": str,
            "lines_added": int,
            "lines_removed": int, 
            "files_count": int,
            "detected_language": str
        }
    
    Returns:
        {
            "commit_type": str,
            "confidence": float,
            "all_probabilities": dict,
            "input_features": dict
        }
    """
    try:
        multifusion_v2 = MultiFusionV2Service()
        
        if not multifusion_v2.is_model_available():
            raise HTTPException(status_code=503, detail="MultiFusion V2 model not available")
        
        # Extract required fields
        commit_message = commit_data.get("commit_message", "")
        lines_added = commit_data.get("lines_added", 0)
        lines_removed = commit_data.get("lines_removed", 0)
        files_count = commit_data.get("files_count", 1)
        detected_language = commit_data.get("detected_language", "unknown_language")

        if not commit_message:
            raise HTTPException(status_code=400, detail="commit_message is required")

        result = multifusion_v2.predict_commit_type(
            commit_message, lines_added, lines_removed, files_count, detected_language
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing commit with MultiFusion V2: {str(e)}")

@router.get("/{repo_id}/members/{member_login}/commits-v2")
async def get_member_commits_analysis_v2(
    repo_id: int,
    member_login: str,
    branch_name: str = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Phân tích commits của member sử dụng MultiFusion V2 model với khả năng advanced
    
    Args:
        repo_id: ID của repository
        member_login: Username của member
        branch_name: Tên branch (optional)
        limit: Số lượng commits tối đa
    
    Returns:
        Phân tích chi tiết với MultiFusion V2 insights
    """
    try:
        service = MemberAnalysisService(db)
        multifusion_v2 = MultiFusionV2Service()
        
        if not multifusion_v2.is_model_available():
            raise HTTPException(status_code=503, detail="MultiFusion V2 model not available")
        
        # Get member commits data using existing method
        commits_analysis = service.get_member_commits_with_analysis(repo_id, member_login, limit, branch_name)

        if not commits_analysis or not isinstance(commits_analysis, dict):
            return {
                "success": True,
                "repository_id": repo_id,
                "member_login": member_login,
                "branch_filter": branch_name,
                "message": "No commits data available",
                "analysis": {},
                "commits": []
            }

        if not commits_analysis.get('commits'):
            return {
                "success": True,
                "repository_id": repo_id,
                "member_login": member_login,
                "branch_filter": branch_name,
                "message": "No commits found for this member",
                "analysis": {},
                "commits": []
            }

        # Convert existing commit data to format expected by MultiFusion V2
        formatted_commits = []
        for commit in commits_analysis['commits']:
            # Skip commits with empty messages
            if not commit.get('message'):
                continue

            # Convert nullable fields to their default values - ưu tiên stats trước
            lines_added = commit.get('insertions') or commit.get('stats', {}).get('insertions', 0)
            if lines_added is None:
                lines_added = 0

            lines_removed = commit.get('deletions') or commit.get('stats', {}).get('deletions', 0)
            if lines_removed is None:
                lines_removed = 0

            files_count = commit.get('files_changed') or commit.get('stats', {}).get('files_changed', 1)
            if files_count is None:
                files_count = 1

            formatted_commits.append({
                'id': commit.get('sha', ''),
                'message': commit.get('message', ''),
                'date': commit.get('date', ''),
                'lines_added': lines_added,
                'lines_removed': lines_removed,
                'files_count': files_count,
                'detected_language': commit.get('language', 'python')  # Use language if available or default to python
            })
            
        # Debug log để kiểm tra dữ liệu truyền vào MultiFusion V2
        print(f"DEBUG: Formatted {len(formatted_commits)} commits for MultiFusion V2")
        if formatted_commits:
            sample_commit = formatted_commits[0]
            print(f"DEBUG: Sample commit data: lines_added={sample_commit['lines_added']}, lines_removed={sample_commit['lines_removed']}, files_count={sample_commit['files_count']}")

        # Skip analysis if no valid commits
        if not formatted_commits:
            return {
                "success": True,
                "repository_id": repo_id,
                "member_login": member_login,
                "branch_filter": branch_name,
                "message": "No valid commits available for analysis",
                "analysis": {},
                "commits": []
            }

        # Analyze with MultiFusion V2
        analysis = multifusion_v2.analyze_member_commits(formatted_commits)

        if "error" in analysis:
            raise HTTPException(status_code=500, detail=analysis["error"])

        # BỔ SUNG: trả về danh sách commit gốc (có insertions, deletions, files_changed...)
        return {
            "success": True,
            "repository_id": repo_id,
            "member_login": member_login,
            "branch_filter": branch_name,
            "model_used": "MultiFusion V2",
            "analysis": analysis,
            "commits": commits_analysis['commits']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing member commits with MultiFusion V2: {str(e)}")

@router.post("/{repo_id}/ai/batch-analyze-v2")
async def analyze_commits_batch_v2(
    repo_id: int,
    request_data: dict,
    db: Session = Depends(get_db)
):
    """
    Batch phân tích nhiều commits sử dụng MultiFusion V2
    
    Args:
        repo_id: ID repository
        request_data: {
            "commits": [
                {
                    "message": str,
                    "lines_added": int,
                    "lines_removed": int,
                    "files_count": int,
                    "detected_language": str
                }
            ]
        }
    
    Returns:
        Kết quả phân tích batch
    """
    try:
        multifusion_v2 = MultiFusionV2Service()
        
        if not multifusion_v2.is_model_available():
            raise HTTPException(status_code=503, detail="MultiFusion V2 model not available")
        
        commits = request_data.get("commits", [])
        if not commits:
            raise HTTPException(status_code=400, detail="commits list is required")
        
        results = []
        for commit in commits:
            result = multifusion_v2.predict_commit_type(
                commit.get("message", ""),
                commit.get("lines_added", 0),
                commit.get("lines_removed", 0),
                commit.get("files_count", 1),
                commit.get("detected_language", "unknown_language")
            )
            
            results.append({
                "input": commit,
                "prediction": result
            })
        
        return {
            "success": True,
            "repository_id": repo_id,
            "model_used": "MultiFusion V2",
            "total_analyzed": len(results),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in MultiFusion V2 batch analysis: {str(e)}")

@router.get("/{repo_id}/ai/model-v2-status")
async def get_multifusion_v2_status(repo_id: int):
    """
    Kiểm tra trạng thái và thông tin của MultiFusion V2 model
    
    Returns:
        Thông tin chi tiết về model V2
    """
    try:
        multifusion_v2 = MultiFusionV2Service()
        model_info = multifusion_v2.get_model_info()
        
        return {
            "success": True,
            "repository_id": repo_id,
            "model_info": model_info,
            "capabilities": {
                "semantic_analysis": "Advanced BERT-based understanding",
                "code_metrics": "Lines added/removed, files count",
                "language_detection": "Programming language integration", 
                "fusion_architecture": "Multi-modal feature fusion",
                "commit_classification": "11 commit types supported",
                "confidence_scoring": "Probabilistic predictions"
            },
            "endpoints": {
                "single_commit": f"/api/repositories/analyze-multifusion-v2-commit",
                "member_analysis": f"/api/repositories/{repo_id}/members/{{member_login}}/commits-v2",
                "batch_analysis": f"/api/repositories/{repo_id}/ai/batch-analyze-v2"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "repository_id": repo_id,
            "error": str(e),
            "model_info": {
                "is_available": False,
                "error_details": str(e)
            }
        }

@router.get("/{repo_id}/ai/compare-models")
async def compare_ai_models(
    repo_id: int,
    commit_message: str,
    lines_added: int = 10,
    lines_removed: int = 5,
    files_count: int = 2,
    detected_language: str = "python"
):
    """
    So sánh kết quả phân tích từ các AI models khác nhau
    
    Args:
        repo_id: ID repository
        commit_message: Commit message để test
        lines_added: Số lines added (default: 10)
        lines_removed: Số lines removed (default: 5)
        files_count: Số files (default: 2)
        detected_language: Ngôn ngữ (default: python)
    
    Returns:
        So sánh kết quả từ các models
    """
    try:
        results = {}
        
        # Test MultiFusion V1 (original)
        try:
            multifusion_v1 = MultiFusionAIService()
            v1_result = multifusion_v1.predict_commit_type(
                commit_message, lines_added, lines_removed, files_count, detected_language
            )
            results["multifusion_v1"] = {
                "success": True,
                "result": {"commit_type": v1_result}
            }
        except Exception as e:
            results["multifusion_v1"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test MultiFusion V2
        try:
            multifusion_v2 = MultiFusionV2Service()
            v2_result = multifusion_v2.predict_commit_type(
                commit_message, lines_added, lines_removed, files_count, detected_language
            )
            results["multifusion_v2"] = {
                "success": True,
                "result": v2_result
            }
        except Exception as e:
            results["multifusion_v2"] = {
                "success": False,
                "error": str(e)
            }
        
        return {
            "success": True,
            "repository_id": repo_id,
            "test_input": {
                "commit_message": commit_message,
                "lines_added": lines_added,
                "lines_removed": lines_removed,
                "files_count": files_count,
                "detected_language": detected_language
            },
            "model_comparison": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing models: {str(e)}")
