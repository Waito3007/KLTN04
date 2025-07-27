from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from db.database import get_db
from services.member_analysis_service import MemberAnalysisService

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