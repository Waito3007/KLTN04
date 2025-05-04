# backend/api/routes/github.py
from fastapi import APIRouter, Request, HTTPException
import httpx
from services.repo_service import get_repo_data
from services.commit_service import save_repo_commits
from services.repo_service import get_repo_id_by_owner_and_name
from services.user_service import get_user_id_by_github_username
from services.branch_service import save_branch
from sqlalchemy.future import select
from fastapi import APIRouter, Depends
from services.repo_service import save_repository
from datetime import datetime
from sqlalchemy import select
from db.models.commits import commits
from db.models.repositories import repositories  # để lấy access token
from schemas.commit import CommitCreate  # schema
from services.github_service import fetch_commits  # hàm gọi GitHub API
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.commit import CommitOut
from db.database import database

from services.branch_service import save_branches
from services.commit_service import save_repo_commits
from services.issue_service import save_issues
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

def get_db():
    return database
# Lấy danh sách commit từ database
@github_router.get("/github/{owner}/{repo}/commits/db")
async def get_commits_from_db(owner: str, repo: str, db: AsyncSession = Depends(get_db)):
    repo_id = await get_repo_id_by_owner_and_name(owner, repo)
    if not repo_id:
        raise HTTPException(status_code=404, detail="Repository not found")

    query = select(commits).where(commits.c.repo_id == repo_id)
    result = await db.fetch_all(query)
    return result

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

    # Lấy commit từ GitHub API
    async with httpx.AsyncClient() as client:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits?sha={branch}"
        headers = {"Authorization": token}
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        commit_list = resp.json()

    # Lưu commit vào database
    repo_id = await get_repo_id_by_owner_and_name(owner, repo)
    if not repo_id:
        raise HTTPException(status_code=404, detail="Repository not found")

    for commit in commit_list:
        commit_data = {
            "sha": commit["sha"],
            "message": commit["commit"]["message"],
            "author_name": commit["commit"]["author"]["name"],
            "author_email": commit["commit"]["author"]["email"],
            "date": commit["commit"]["author"]["date"],
            "repo_id": repo_id,
        }
        await save_commit(commit_data)

    return {"message": "Commits saved successfully!"}
#lưu branchbranch vào database
@github_router.post("/github/{owner}/{repo}/save-branches")
async def save_branches(owner: str, repo: str, request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    # Lấy danh sách branch từ GitHub API
    async with httpx.AsyncClient() as client:
        url = f"https://api.github.com/repos/{owner}/{repo}/branches"
        headers = {"Authorization": token}
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        branches = resp.json()

    # Lưu branch vào database
    repo_id = await get_repo_id_by_owner_and_name(owner, repo)
    if not repo_id:
        raise HTTPException(status_code=404, detail="Repository not found")

    for branch in branches:
        branch_data = {
            "name": branch["name"],
            "repo_id": repo_id,
        }
        await save_branch(branch_data)

    return {"message": "Branches saved successfully!"}
# lưu issues vào database
@github_router.post("/github/{owner}/{repo}/save-issues")
async def save_issues(owner: str, repo: str, request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    # Lấy danh sách issue từ GitHub API
    async with httpx.AsyncClient() as client:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        headers = {"Authorization": token}
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        issues = resp.json()

    
    # Lưu issue vào database
    repo_id = await get_repo_id_by_owner_and_name(owner, repo)
    if not repo_id:
        raise HTTPException(status_code=404, detail="Repository not found")

    for issue in issues:
        issue_data = {
            "title": issue["title"],
            "body": issue["body"],
            "state": issue["state"],
            "created_at": issue["created_at"],
            "updated_at": issue["updated_at"],
            "repo_id": repo_id,
        }
        await save_issue(issue_data)

    return {"message": "Issues saved successfully!"}
#save repo vào database
async def save_repo(owner: str, repo: str, request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    async with httpx.AsyncClient() as client:
        url = f"https://api.github.com/repos/{owner}/{repo}"
        headers = {"Authorization": token}
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        repo_data = resp.json()

    repo_entry = {
        "github_id": repo_data["id"],
        "name": repo_data["name"],
        "owner": repo_data["owner"]["login"],
        "description": repo_data["description"],
        "stars": repo_data["stargazers_count"],
        "forks": repo_data["forks_count"],
        "language": repo_data["language"],
        "open_issues": repo_data["open_issues_count"],
        "url": repo_data["html_url"],
    }

    try:
        await save_repository(repo_entry)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving repository: {str(e)}")

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
# đồng bộ toàn bộ dữ liệu
@github_router.post("/github/{owner}/{repo}/sync-all")
async def sync_all(owner: str, repo: str, request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    # Đồng bộ repository
    await save_repo(owner, repo, request)

    # Đồng bộ branches
    await save_branches(owner, repo, request)

    # Đồng bộ commits
    await save_repo_commits(owner, repo, request)

    # Đồng bộ issues
    await save_issues(owner, repo, request)

    return {"message": "Đồng bộ toàn bộ dữ liệu thành công!"}