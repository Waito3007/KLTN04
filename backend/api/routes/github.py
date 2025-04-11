# backend/api/routes/github.py
from fastapi import APIRouter, Request, HTTPException
import httpx
from services.repo_service import get_repo_data
from services.commit_service import save_commit
from services.repo_service import get_repo_id_by_owner_and_name
from services.user_service import get_user_id_by_github_username
from sqlalchemy.future import select
from fastapi import APIRouter, Depends

from datetime import datetime
from sqlalchemy import select
from db.models.commits import commits
from db.models.repositories import repositories  # để lấy access token
from schemas.commit import CommitCreate  # schema
from services.github_service import fetch_commits  # hàm gọi GitHub API
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.commit import CommitOut
from db.database import database
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

@github_router.get("/github/{owner}/{repo}/commits")
async def get_commits(owner: str, repo: str, request: Request, branch: str = "main"):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    async with httpx.AsyncClient() as client:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits?sha={branch}"
        headers = {"Authorization": token}

        resp = await client.get(url, headers=headers)
        if resp.status_code == 409:
            return []
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        return resp.json()

@github_router.get("/github/{owner}/{repo}/branches")
async def get_branches(owner: str, repo: str, request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    async with httpx.AsyncClient() as client:
        url = f"https://api.github.com/repos/{owner}/{repo}/branches"
        headers = {"Authorization": token}

        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        return resp.json()

@github_router.post("/github/{owner}/{repo}/save-commits")
async def save_repo_commits(owner: str, repo: str, request: Request, branch: str = "main"):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    # Lấy commit từ GitHub
    async with httpx.AsyncClient() as client:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits?sha={branch}"
        headers = {"Authorization": token}
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        
        commit_list = resp.json()

    # Giả sử đã có user_id và repo_id (em có thể truyền vào hoặc ánh xạ theo repo)
    for commit in commit_list:
        commit_data = {
            "sha": commit["sha"],
            "message": commit["commit"]["message"],
            "author_id": None,  # TODO: ánh xạ user GitHub nếu có
            "repo_id": None,    # TODO: ánh xạ repo nếu đã lưu vào DB
            "committed_at": commit["commit"]["author"]["date"],
            "insertions": 0,    # TODO: lấy từ GitHub nếu cần detail
            "deletions": 0,
            "files_changed": 0
        }
        await save_commit(commit_data)

    return {"message": "Commits saved successfully!"}



@github_router.post("/github/{owner}/{repo}/save-commits")
async def save_repo_commits(owner: str, repo: str, request: Request, branch: str = "main"):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    async with httpx.AsyncClient() as client:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits?sha={branch}"
        headers = {"Authorization": token}
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        
        commit_list = resp.json()

    repo_id = await get_repo_id_by_owner_and_name(owner, repo)
    if not repo_id:
        raise HTTPException(status_code=404, detail="Repository not found in database")

    for commit in commit_list:
        gh_author = commit["author"]["login"] if commit.get("author") else None
        author_id = await get_user_id_by_github_username(gh_author) if gh_author else None

        commit_data = {
            "sha": commit["sha"],
            "message": commit["commit"]["message"],
            "author_id": author_id,
            "repo_id": repo_id,
            "committed_at": commit["commit"]["author"]["date"],
            "insertions": 0,
            "deletions": 0,
            "files_changed": 0
        }
        await save_commit(commit_data)

    return {"message": "Commits saved successfully!"}


def get_db():
    return database

@github_router.get("/commits")
async def get_commits(db = Depends(get_db)):
    query = commits.select()
    result = await db.fetch_all(query)
    return result

# Thêm vào cuối file github.py

@github_router.get("/sync-commits")
async def sync_commits(
    repo_id: int,
    branch: str = "main",
    since: str = None,
    until: str = None,
    db: AsyncSession = Depends(get_db)
):
    # 1. Lấy repo từ DB
    repo = await db.scalar(select(Repository).where(Repository.id == repo_id))
    if not repo:
        raise HTTPException(status_code=404, detail="Repository không tồn tại")

    # 2. Gọi API GitHub lấy commit theo filter
    commits_data = await fetch_commits(
        token=repo.token,
        owner=repo.owner,
        name=repo.name,
        branch=branch,
        since=since,
        until=until
    )

    # 3. Lưu commit vào DB (nếu chưa có)
    new_commits = []
    for item in commits_data:
        sha = item["sha"]
        # Check trùng
        existing = await db.scalar(select(Commit).where(Commit.sha == sha))
        if existing:
            continue

        new_commit = CommitCreate(
            sha=sha,
            message=item["commit"]["message"],
            author=item["commit"]["author"]["name"],
            date=item["commit"]["author"]["date"],
            repository_id=repo.id
        )
        commit_obj = Commit(**new_commit.dict())
        db.add(commit_obj)
        new_commits.append(commit_obj)

    await db.commit()

    return {
        "message": f"Đồng bộ thành công {len(new_commits)} commit.",
        "data": [c.sha for c in new_commits]
    }
