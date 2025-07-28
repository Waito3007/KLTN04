# backend/api/routes/branch.py
from fastapi import APIRouter, Request, HTTPException
import httpx
from services.branch_service import save_branch
from services.repo_service import get_repo_id_by_owner_and_name

branch_router = APIRouter()

# API lấy danh sách branch từ DB theo repoId
from db.models.branches import branches as branches_table
from db.models.repositories import repositories as repositories_table
from sqlalchemy import select
from db.database import database

@branch_router.get("/{repo_id}/branches")
async def get_branches_db(repo_id: int):
    # Trả về danh sách branch của repo đã lưu trong DB
    query = select([
        branches_table.c.id,
        branches_table.c.name,
        branches_table.c.sha,
        branches_table.c.is_default,
        branches_table.c.is_protected,
        branches_table.c.created_at,
        branches_table.c.last_commit_date,
        branches_table.c.commits_count,
        branches_table.c.contributors_count
    ]).where(branches_table.c.repo_id == repo_id)
    result = await database.fetch_all(query)
    branches = [
        {
            "id": row["id"],
            "name": row["name"],
            "sha": row["sha"],
            "is_default": row["is_default"],
            "is_protected": row["is_protected"],
            "created_at": row["created_at"],
            "last_commit_date": row["last_commit_date"],
            "commits_count": row["commits_count"],
            "contributors_count": row["contributors_count"]
        }
        for row in result
    ]
    return {"branches": branches}

@branch_router.get("/github/{owner}/{repo}/branches")
async def get_branches(owner: str, repo: str, request: Request):
    # Lấy token từ header
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    # Gọi GitHub API lấy danh sách branch
    async with httpx.AsyncClient() as client:
        url = f"https://api.github.com/repos/{owner}/{repo}/branches"
        headers = {"Authorization": token}

        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        return resp.json()
