from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from core.security import get_current_user
from db.database import get_db
from interfaces.service_factory import get_task_commit_service
from interfaces.task_commit_service import ITaskCommitService

router = APIRouter()

class TaskCommitLinkResponse(BaseModel):
    success: bool
    message: str
    commits_found: int
    commits: List[Dict[str, Any]]

class CommitLinkRequest(BaseModel):
    commit_shas: Optional[List[str]] = None

class CommitInfo(BaseModel):
    sha: str
    message: str
    author_name: str
    author_email: str
    committed_date: str
    insertions: int
    deletions: int
    files_changed: int
    url: str

@router.post("/projects/{owner}/{repo}/tasks/{task_id}/link-commits", response_model=TaskCommitLinkResponse)
async def link_commits_to_task(
    owner: str,
    repo: str,
    task_id: int,
    request: CommitLinkRequest = Body(default=None),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
    task_commit_service: ITaskCommitService = Depends(get_task_commit_service)
):
    """
    Liên kết commits với task - tự động hoặc thủ công
    Nếu có commit_shas thì liên kết thủ công, nếu không thì tự động
    """
    try:
        if request and request.commit_shas:
            # Liên kết thủ công với danh sách commit SHA đã chọn
            result = await task_commit_service.link_specific_commits(task_id, owner, repo, request.commit_shas)
        else:
            # Liên kết tự động dựa trên pattern matching
            result = await task_commit_service.link_commits_to_task(task_id, owner, repo)
        return TaskCommitLinkResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to link commits: {str(e)}")

@router.post("/projects/{owner}/{repo}/tasks/{task_id}/commits", response_model=TaskCommitLinkResponse)
async def link_commits_to_task_shorthand(
    owner: str,
    repo: str,
    task_id: int,
    request: CommitLinkRequest = Body(default=None),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
    task_commit_service: ITaskCommitService = Depends(get_task_commit_service)
):
    """
    Liên kết commits với task (shorthand endpoint) - tự động hoặc thủ công
    Nếu có commit_shas thì liên kết thủ công, nếu không thì tự động
    """
    try:
        if request and request.commit_shas:
            # Liên kết thủ công với danh sách commit SHA đã chọn
            result = await task_commit_service.link_specific_commits(task_id, owner, repo, request.commit_shas)
        else:
            # Liên kết tự động dựa trên pattern matching
            result = await task_commit_service.link_commits_to_task(task_id, owner, repo)
        return TaskCommitLinkResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to link commits: {str(e)}")

@router.get("/projects/{owner}/{repo}/tasks/{task_id}/commits", response_model=List[CommitInfo])
async def get_task_related_commits(
    owner: str,
    repo: str,
    task_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
    task_commit_service: ITaskCommitService = Depends(get_task_commit_service)
):
    """
    Lấy danh sách commits liên quan đến task
    """
    try:
        commits = await task_commit_service.get_task_related_commits(task_id)
        return [CommitInfo(**commit) for commit in commits]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task commits: {str(e)}")

@router.get("/projects/{owner}/{repo}/users/{username}/commits", response_model=Dict[str, Any])
async def get_user_recent_commits(
    owner: str,
    repo: str,
    username: str,
    limit: int = Query(10, description="Number of commits per page"),
    offset: int = Query(0, description="Number of commits to skip"),
    search: str = Query(None, description="Search query for commit messages, SHA, or author"),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
    task_commit_service: ITaskCommitService = Depends(get_task_commit_service)
):
    """
    Lấy commits của user trong repository với hỗ trợ tìm kiếm và phân trang
    """
    try:
        result = await task_commit_service.get_user_commits_with_pagination(
            owner, repo, username, limit, offset, search
        )
        return {
            "commits": [CommitInfo(**commit) for commit in result.get("commits", [])],
            "total": result.get("total", 0),
            "has_more": result.get("has_more", False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user commits: {str(e)}")

@router.get("/projects/{owner}/{repo}/authors")
async def get_repository_authors(
    owner: str,
    repo: str,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
    task_commit_service: ITaskCommitService = Depends(get_task_commit_service)
):
    """
    Lấy danh sách tác giả commits trong repository với thống kê
    """
    try:
        authors = await task_commit_service.get_repository_authors(owner, repo)
        return {
            "success": True,
            "authors": authors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get repository authors: {str(e)}")

@router.get("/projects/{owner}/{repo}/search-commits")
async def search_commits_by_pattern(
    owner: str,
    repo: str,
    username: str = Query(..., description="GitHub username"),
    task_title: str = Query(..., description="Task title for pattern matching"),
    days_back: int = Query(30, description="Days to look back for commits"),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
    task_commit_service: ITaskCommitService = Depends(get_task_commit_service)
):
    """
    Tìm kiếm commits theo pattern matching với task title
    """
    try:
        commits = await task_commit_service.search_commits_by_pattern(
            owner, repo, username, task_title, days_back
        )
        return {
            "success": True,
            "commits_found": len(commits),
            "commits": commits
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search commits: {str(e)}")
