# backend/api/routes/commit.py
from fastapi import APIRouter, Request, HTTPException, Depends
import httpx
from services.commit_service import save_commit
from services.repo_service import get_repo_id_by_owner_and_name
from services.github_service import fetch_commits
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from db.models.commits import commits
from db.models.repositories import repositories
from db.database import database
from schemas.commit import CommitCreate, CommitOut

commit_router = APIRouter()

def get_db():
    return database

# Endpoint lấy danh sách commit của một repository
@commit_router.get("/github/{owner}/{repo}/commits")
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

# Lấy danh sách commit từ database
@commit_router.get("/github/{owner}/{repo}/commits/db")
async def get_commits_from_db(owner: str, repo: str, db: AsyncSession = Depends(get_db)):
    repo_id = await get_repo_id_by_owner_and_name(owner, repo)
    if not repo_id:
        raise HTTPException(status_code=404, detail="Repository not found")

    query = select(commits).where(commits.c.repo_id == repo_id)
    result = await db.fetch_all(query)
    return result

# Endpoint lưu commit vào database
@commit_router.post("/github/{owner}/{repo}/save-commits")
async def save_repo_commits(owner: str, repo: str, request: Request, branch: str = None, limit: int = 50):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    # Nếu không truyền branch, lấy branch mặc định từ GitHub
    if not branch:
        async with httpx.AsyncClient() as client:
            repo_url = f"https://api.github.com/repos/{owner}/{repo}"
            headers = {"Authorization": token}
            repo_resp = await client.get(repo_url, headers=headers)
            if repo_resp.status_code != 200:
                raise HTTPException(status_code=repo_resp.status_code, detail=repo_resp.text)
            repo_data = repo_resp.json()
            branch = repo_data.get("default_branch", "main")

    # Lấy danh sách commit từ GitHub API (giới hạn số lượng)
    async with httpx.AsyncClient() as client:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits?sha={branch}&per_page={limit}"
        headers = {"Authorization": token}
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        commit_list = resp.json()

    # Lấy repo_id từ cơ sở dữ liệu
    repo_id = await get_repo_id_by_owner_and_name(owner, repo)
    if not repo_id:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Lưu từng commit vào cơ sở dữ liệu (song song để tăng tốc)
    saved_commits = 0
    async with httpx.AsyncClient() as client:
        for commit in commit_list[:limit]:  # Giới hạn thêm một lần nữa
            try:
                # Lấy thông tin chi tiết của commit
                commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit['sha']}"
                commit_resp = await client.get(commit_url, headers={"Authorization": token})
                if commit_resp.status_code != 200:
                    continue  # Bỏ qua commit nếu không lấy được thông tin chi tiết

                commit_details = commit_resp.json()
                stats = commit_details.get("stats", {})
                commit_data = {
                    "sha": commit["sha"],
                    "message": commit["commit"]["message"],
                    "author_name": commit["commit"]["author"]["name"],
                    "author_email": commit["commit"]["author"]["email"],
                    "date": datetime.strptime(commit["commit"]["author"]["date"], "%Y-%m-%dT%H:%M:%SZ"),
                    "insertions": stats.get("additions", 0),
                    "deletions": stats.get("deletions", 0),
                    "files_changed": stats.get("total", 0),
                    "repo_id": repo_id,
                }
                await save_commit(commit_data)
                saved_commits += 1
            except Exception as e:
                print(f"Lỗi khi lưu commit {commit['sha']}: {e}")
                continue

    return {"message": f"Đã lưu {saved_commits}/{len(commit_list)} commits!"}

# Endpoint lấy tất cả commit từ database
@commit_router.get("/commits")
async def get_all_commits(db = Depends(get_db)):
    query = commits.select()  # Lấy tất cả commit
    result = await db.fetch_all(query)
    return result

# Endpoint đồng bộ commit từ GitHub về database
@commit_router.get("/sync-commits")
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
        "data": [c.sha for c in new_commits]
    }
