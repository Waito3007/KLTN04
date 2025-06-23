from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any
from db.database import get_db
from core.security import get_current_user, CurrentUser

router = APIRouter(prefix="/api", tags=["repositories"])

@router.get("/repositories")
async def get_repositories(
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Lấy danh sách repositories của user - REQUIRE AUTHENTICATION"""
    try:
        # Chỉ user đã đăng nhập mới được lấy repos với avatar của owner
        query = text("""
            SELECT DISTINCT
                r.id,
                r.name,
                r.owner,
                r.full_name,
                r.description,
                r.stars,
                r.forks,
                r.language,
                r.is_private,
                r.url,
                r.created_at,
                u.avatar_url as owner_avatar_url
            FROM repositories r
            LEFT JOIN repository_collaborators rc ON r.id = rc.repository_id
            LEFT JOIN collaborators c ON rc.collaborator_id = c.id
            LEFT JOIN users u ON r.owner = u.github_username
            WHERE 
                r.owner = :github_username  -- Repos owned by user
                OR c.github_username = :github_username  -- Repos where user is collaborator
            ORDER BY r.name
        """)
        
        result = db.execute(query, {"github_username": current_user.github_username}).fetchall()
        
        repositories = []
        for row in result:
            repo = {
                "id": row[0],
                "name": row[1],
                "owner": {
                    "login": row[2],
                    "avatar_url": row[11]  # owner_avatar_url from JOIN with users table
                },
                "full_name": row[3] or f"{row[2]}/{row[1]}",
                "description": row[4],
                "stargazers_count": row[5] or 0,
                "forks_count": row[6] or 0,
                "language": row[7],
                "private": row[8] or False,
                "html_url": row[9],
                "created_at": row[10].isoformat() if row[10] else None
            }
            repositories.append(repo)
        
        return repositories
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching repositories: {str(e)}")

@router.get("/github/repositories")  
async def get_github_repositories(
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Alias cho compatibility với frontend"""
    return await get_repositories(db, current_user)

@router.get("/{owner}/{repo}/branches")
async def get_repository_branches(
    owner: str,
    repo: str,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Lấy danh sách branches của repository theo owner/repo"""
    try:
        # Get repository first
        repo_query = text("""
            SELECT id FROM repositories 
            WHERE owner = :owner AND name = :repo
        """)
        repo_result = db.execute(repo_query, {"owner": owner, "repo": repo}).fetchone()
        
        if not repo_result:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo} not found")
        
        repo_id = repo_result[0]
        
        # Get branches for this repository
        branches_query = text("""
            SELECT 
                id, name, repo_id, creator_name, last_committer_name,
                sha, is_default, is_protected, created_at, last_commit_date,
                commits_count, contributors_count
            FROM branches 
            WHERE repo_id = :repo_id
            ORDER BY is_default DESC, name ASC
        """)
        
        results = db.execute(branches_query, {"repo_id": repo_id}).fetchall()
        
        branches = []
        for row in results:
            branch = {
                "id": row[0],
                "name": row[1],
                "repo_id": row[2],
                "creator_name": row[3],
                "last_committer_name": row[4],
                "sha": row[5],
                "is_default": row[6],
                "is_protected": row[7],
                "created_at": row[8].isoformat() if row[8] else None,
                "last_commit_date": row[9].isoformat() if row[9] else None,
                "commits_count": row[10] or 0,
                "contributors_count": row[11] or 0
            }
            branches.append(branch)
        
        return {
            "repository": f"{owner}/{repo}",
            "branches": branches,
            "total": len(branches)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching branches: {str(e)}")
