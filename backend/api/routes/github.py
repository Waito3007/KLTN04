# KLTN04\backend\api\routes\github.py
from fastapi import APIRouter
from services.repo_service import get_repo_data

github_router = APIRouter()

@github_router.get("/github/{owner}/{repo}")
async def fetch_repo(owner: str, repo: str):
    return await get_repo_data(owner, repo)
