from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.db.database import get_db
from backend.services.commit_service import get_commit_by_sha, get_commits_by_repo_id
from ..services.commit_classification_service import commit_classification_service

router = APIRouter()

@router.post("/commits/{commit_id}/classify", tags=["AI"])
async def classify_commit_by_id_endpoint(commit_id: int, db: Session = Depends(get_db)):
    """
    Classifies a specific commit by its ID.
    """
    # Lấy tất cả commits để tìm commit theo ID (có thể tối ưu sau)
    from backend.db.models.commits import commits
    from backend.db.database import database
    from sqlalchemy import select
    
    query = select(commits).where(commits.c.id == commit_id)
    commit = await database.fetch_one(query)
    
    if not commit:
        raise HTTPException(status_code=404, detail="Commit not found")

    commit_data = {
        "message": commit.message or "",
        "insertions": commit.insertions or 0,
        "deletions": commit.deletions or 0,
        "files_changed": commit.files_changed or 0,
        "file_types": commit.file_types or {},
    }
    result = commit_classification_service.classify_commit(commit_data)
    return result

@router.post("/commits/{commit_sha}/classify", tags=["AI"])
async def classify_commit_endpoint(commit_sha: str, db: Session = Depends(get_db)):
    """
    Classifies a specific commit by its SHA.
    """
    commit = await get_commit_by_sha(commit_sha)
    if not commit:
        raise HTTPException(status_code=404, detail="Commit not found")

    commit_data = {
        "message": commit.get("message", ""),
        "insertions": commit.get("insertions", 0),
        "deletions": commit.get("deletions", 0),
        "files_changed": commit.get("files_changed", 0),
        "file_types": commit.get("file_types", {}),
    }
    result = commit_classification_service.classify_commit(commit_data)
    return result
