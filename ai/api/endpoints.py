from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.db.database import get_db
from backend.services.commit_service import commit_service
from ai.services.commit_classification_service import commit_classification_service

router = APIRouter()

@router.post("/commits/{commit_id}/classify", tags=["AI"])
def classify_commit_endpoint(commit_id: int, db: Session = Depends(get_db)):
    """
    Classifies a specific commit by its ID.
    """
    db_commit = commit_service.get_commit(db, commit_id=commit_id)
    if db_commit is None:
        raise HTTPException(status_code=404, detail="Commit not found")

    # Convert the SQLAlchemy model object to a dictionary
    commit_data = {
        "message": db_commit.message,
        "insertions": db_commit.insertions,
        "deletions": db_commit.deletions,
        "files_changed": db_commit.files_changed,
        "file_types": db_commit.file_types, # Assuming this field exists and is relevant
    }

    result = commit_classification_service.classify_commit(commit_data)
    return result
