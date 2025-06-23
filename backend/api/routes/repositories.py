from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any
from db.database import get_db

router = APIRouter(prefix="/api", tags=["repositories"])

@router.get("/repositories")
async def get_repositories(db: Session = Depends(get_db)):
    """Lấy danh sách repositories từ database - không cần auth"""
    try:
        query = text("""
            SELECT 
                id,
                name,
                owner,
                full_name,
                description,
                stars,
                forks,
                language,
                is_private,
                url,
                created_at
            FROM repositories
            ORDER BY name
        """)
        
        result = db.execute(query).fetchall()
        
        repositories = []
        for row in result:
            repo = {
                "id": row[0],
                "name": row[1],
                "owner": {
                    "login": row[2]
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
async def get_github_repositories(db: Session = Depends(get_db)):
    """Alias cho compatibility với frontend"""
    return await get_repositories(db)
