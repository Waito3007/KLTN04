# backend/api/routes/contributors.py
from fastapi import APIRouter, Depends, HTTPException, status, Header
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from core.security import get_current_user
from services.collaborator_service import (
    get_collaborators_with_fallback,
    sync_repository_collaborators,
    get_collaborators_by_repo
)
from db.models.repositories import repositories
from db.database import database
from sqlalchemy import select

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/{owner}/{repo}")
async def get_repository_collaborators(
    owner: str,
    repo: str,
    authorization: Optional[str] = Header(None),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get collaborators for a specific repository"""
    try:
        logger.info(f"Getting collaborators for repository {owner}/{repo}")
          # Get repository ID first
        repo_query = select(repositories.c.id).where(
            (repositories.c.owner == owner) &
            (repositories.c.name == repo)
        )
        repo_result = await database.fetch_one(repo_query)
        
        if not repo_result:
            logger.warning(f"Repository {owner}/{repo} not found in database")
            # Return empty but valid response
            return {
                "repository": f"{owner}/{repo}",
                "collaborators": [],
                "count": 0,
                "has_synced_data": False,
                "message": "Repository not found in database. Please sync first."
            }
        
        # Only get from database - NO automatic fallback
        collaborators = await get_collaborators_by_repo(repo_result.id)
        has_synced_data = len(collaborators) > 0
          # Create response in expected format
        response = {
            "repository": f"{owner}/{repo}",
            "collaborators": collaborators,
            "count": len(collaborators),
            "has_synced_data": has_synced_data,
            "message": (
                f"Loaded {len(collaborators)} synced collaborators from database" if has_synced_data 
                else "No collaborators found in database. Click 'Sync' to import from GitHub."
            )
        }
        
        logger.info(f"Retrieved {len(collaborators)} collaborators for {owner}/{repo}")
        return response
        
    except Exception as e:
        logger.error(f"Error getting collaborators for {owner}/{repo}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collaborators: {str(e)}"
        )

@router.post("/{owner}/{repo}/sync")
async def sync_repository_collaborators_endpoint(
    owner: str,
    repo: str,
    authorization: Optional[str] = Header(None),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Sync collaborators from GitHub to database"""
    try:
        logger.info(f"Syncing collaborators for repository {owner}/{repo}")
        
        # Extract GitHub token from Authorization header
        github_token = None
        if authorization:
            # Handle both "Bearer token" and "token token" formats
            if authorization.startswith("Bearer "):
                github_token = authorization[7:]
            elif authorization.startswith("token "):
                github_token = authorization[6:]
            else:
                github_token = authorization
        
        if not github_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="GitHub access token required for syncing collaborators"
            )
        
        # Ensure repository exists in our database
        repo_query = select(repositories.c.id).where(
            (repositories.c.owner == owner) &
            (repositories.c.name == repo)
        )
        repo_result = await database.fetch_one(repo_query)
        
        if not repo_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Repository {owner}/{repo} not found in database"
            )
        
        # Sync collaborators using the service
        sync_result = await sync_repository_collaborators(
            owner=owner,
            repo=repo,
            github_token=github_token
        )
        
        if sync_result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Sync failed: {sync_result.get('error', 'Unknown error')}"
            )
        
        logger.info(f"Successfully synced collaborators for {owner}/{repo}")
        return sync_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing collaborators for {owner}/{repo}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync collaborators: {str(e)}"
        )

@router.get("/repository/{repo_id}")
async def get_collaborators_by_repository_id(
    repo_id: int,
    current_user: dict = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get collaborators by repository ID"""
    try:
        logger.info(f"Getting collaborators for repository ID {repo_id}")
        
        collaborators = await get_collaborators_by_repo(repo_id)
        
        logger.info(f"Retrieved {len(collaborators)} collaborators for repository ID {repo_id}")
        return collaborators
        
    except Exception as e:
        logger.error(f"Error getting collaborators for repository ID {repo_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collaborators: {str(e)}"
        )

@router.get("/health")
async def collaborators_health_check():
    """Health check endpoint for collaborators API"""
    return {
        "status": "healthy",
        "service": "collaborators",
        "timestamp": datetime.now().isoformat()
    }