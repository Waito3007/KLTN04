# backend/api/routes/branch.py
from fastapi import APIRouter, Request, HTTPException
import httpx
from services.branch_service import save_branch
from services.repo_service import get_repo_id_by_owner_and_name

branch_router = APIRouter()

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
