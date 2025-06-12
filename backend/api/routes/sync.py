# backend/api/routes/sync.py
from fastapi import APIRouter, Request, HTTPException
import httpx
from services.repo_service import save_repository, get_repo_id_by_owner_and_name
from services.branch_service import save_branch
from services.commit_service import save_commit
from services.issue_service import save_issue

sync_router = APIRouter()

# Đồng bộ toàn bộ dữ liệu
@sync_router.post("/github/{owner}/{repo}/sync-all")
async def sync_all(owner: str, repo: str, request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    try:
        # Đồng bộ repository
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
        await save_repository(repo_entry)        # Đồng bộ branches
        async with httpx.AsyncClient() as client:
            url = f"https://api.github.com/repos/{owner}/{repo}/branches"
            headers = {"Authorization": token}
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)

            branches_data = resp.json()

        # Lấy repo_id để save branches
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository not found")

        for branch_data in branches_data:
            branch_entry = {
                "name": branch_data["name"],
                "repo_id": repo_id,
            }
            await save_branch(branch_entry)

        return {"message": f"Đồng bộ repository {owner}/{repo} thành công!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đồng bộ {owner}/{repo}: {str(e)}")

# Endpoint đồng bộ nhanh - chỉ thông tin cơ bản
@sync_router.post("/github/{owner}/{repo}/sync-basic")
async def sync_basic(owner: str, repo: str, request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    try:
        # Chỉ đồng bộ repository
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
        await save_repository(repo_entry)
        
        return {"message": f"Đồng bộ cơ bản {owner}/{repo} thành công!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đồng bộ cơ bản {owner}/{repo}: {str(e)}")
