# backend/api/routes/github.py
# Import các thư viện và module cần thiết
from fastapi import APIRouter, Request, HTTPException  # Framework web và xử lý request
import httpx  # Thư viện HTTP client async
from services.repo_service import get_repo_data  # Service xử lý repository
# from services.commit_service import save_commit  # Service lưu commit
from services.repo_service import get_repo_id_by_owner_and_name  # Lấy ID repo theo tên
from services.user_service import get_user_id_by_github_username  # Lấy ID user theo GitHub username
from sqlalchemy.future import select  # Câu lệnh SQL select async
from fastapi import APIRouter, Depends  # Dependency injection

from datetime import datetime  # Xử lý thời gian
from sqlalchemy import select  # Câu lệnh SQL select
from db.models.commits import commits  # Model bảng commits
from db.models.repositories import repositories  # Model bảng repositories
from schemas.commit import CommitCreate  # Schema tạo commit
from services.github_service import fetch_commits  # Service gọi GitHub API
from sqlalchemy.ext.asyncio import AsyncSession  # Session database async
from schemas.commit import CommitOut  # Schema trả về commit
from db.database import database  # Kết nối database

# Khởi tạo router cho các endpoint GitHub
github_router = APIRouter()

# Endpoint lấy thông tin repository cụ thể
@github_router.get("/github/{owner}/{repo}")
async def fetch_repo(owner: str, repo: str):
    return await get_repo_data(owner, repo)

# Endpoint lấy danh sách repository của user
@github_router.get("/github/repos")
async def get_user_repos(request: Request):
    # Lấy token từ header Authorization
    token = request.headers.get("Authorization")
    
    # Kiểm tra token hợp lệ (phải bắt đầu bằng "token ")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    # Gọi GitHub API để lấy danh sách repo
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.github.com/user/repos",
            headers={"Authorization": token}
        )
        # Nếu lỗi thì raise exception
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    
    # Trả về kết quả dạng JSON
    return resp.json()

# Endpoint lấy danh sách commit của một repository
@github_router.get("/github/{owner}/{repo}/commits")
async def get_commits(owner: str, repo: str, request: Request, branch: str = "main"):
    # Lấy token từ header
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    # Gọi GitHub API để lấy commit
    async with httpx.AsyncClient() as client:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits?sha={branch}"
        headers = {"Authorization": token}

        resp = await client.get(url, headers=headers)
        # Xử lý trường hợp repository trống (409)
        if resp.status_code == 409:
            return []
        # Xử lý lỗi khác
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        return resp.json()

# Hàm dependency để lấy database session
def get_db():
    return database

# Endpoint lấy commit từ database
@github_router.get("/github/{owner}/{repo}/commits")
async def get_commits_from_db(owner: str, repo: str, db: AsyncSession = Depends(get_db)):
    # Lấy repo_id từ owner và repo name
    repo_id = await get_repo_id_by_owner_and_name(owner, repo)
    if not repo_id:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Query lấy tất cả commit của repo
    query = select(commits).where(commits.c.repo_id == repo_id)
    result = await db.fetch_all(query)
    return result

# Endpoint lấy danh sách branch của repository
@github_router.get("/github/{owner}/{repo}/branches")
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

# Endpoint lưu commit vào database
@github_router.post("/github/{owner}/{repo}/save-commits")
async def save_repo_commits(owner: str, repo: str, request: Request, branch: str = "main"):
    # Lấy token từ header
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

    # Lấy repo_id từ database
    repo_id = await get_repo_id_by_owner_and_name(owner, repo)
    if not repo_id:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Lưu từng commit vào database
    for commit in commit_list:
        commit_data = {
            "sha": commit["sha"],  # Commit hash
            "message": commit["commit"]["message"],  # Nội dung commit
            "author_name": commit["commit"]["author"]["name"],  # Tên tác giả
            "author_email": commit["commit"]["author"]["email"],  # Email tác giả
            "date": commit["commit"]["author"]["date"],  # Ngày commit
            "repo_id": repo_id,  # ID repository
        }
        await save_commit(commit_data)

    return {"message": "Commits saved successfully!"}

# Dependency để lấy database session
def get_db():
    return database

# Endpoint lấy tất cả commit từ database
@github_router.get("/commits")
async def get_commits(db = Depends(get_db)):
    query = commits.select()  # Lấy tất cả commit
    result = await db.fetch_all(query)
    return result

# Endpoint đồng bộ commit từ GitHub về database
@github_router.get("/sync-commits")
async def sync_commits(
    repo_id: int,
    branch: str = "main",
    since: str = None,
    until: str = None,
    db: AsyncSession = Depends(get_db)
):
    # 1. Lấy thông tin repository từ database
    repo = await db.scalar(select(repositories).where(repositories.c.id == repo_id))
    if not repo:
        raise HTTPException(status_code=404, detail="Repository không tồn tại")

    # 2. Gọi GitHub API lấy commit với các tham số lọc
    commits_data = await fetch_commits(
        token=repo.token,  # Access token
        owner=repo.owner,  # Chủ repository
        name=repo.name,  # Tên repository
        branch=branch,  # Branch cần lấy
        since=since,  # Lọc từ thời gian
        until=until  # Lọc đến thời gian
    )

    # 3. Lưu commit mới vào database
    new_commits = []
    for item in commits_data:
        sha = item["sha"]
        # Kiểm tra commit đã tồn tại chưa
        existing = await db.scalar(select(commits).where(commits.c.sha == sha))
        if existing:
            continue  # Bỏ qua nếu đã tồn tại

        # Tạo commit mới
        new_commit = CommitCreate(
            sha=sha,
            message=item["commit"]["message"],
            author=item["commit"]["author"]["name"],
            date=item["commit"]["author"]["date"],
            repository_id=repo.id
        )
        commit_obj = commits.insert().values(**new_commit.dict())
        await db.execute(commit_obj)
        new_commits.append(new_commit)

    await db.commit()

    return {
        "message": f"Đồng bộ thành công {len(new_commits)} commit.",
        "data": [c.sha for c in new_commits]  # Trả về danh sách SHA commit mới
    }