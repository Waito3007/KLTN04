# backend/api/routes/github.py
from fastapi import APIRouter, Request, HTTPException
import httpx
from services.repo_service import get_repo_data

github_router = APIRouter()

@github_router.get("/github/{owner}/{repo}")
async def fetch_repo(owner: str, repo: str):
    return await get_repo_data(owner, repo)


@github_router.get("/github/repos")
async def get_user_repos(request: Request):
    token = request.headers.get("Authorization")
    
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.github.com/user/repos",
            headers={"Authorization": token}
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    
    return resp.json()

