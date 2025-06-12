# backend/api/routes/repo.py
from fastapi import APIRouter, Request, HTTPException
import httpx
from services.repo_service import get_repo_data, save_repository

repo_router = APIRouter()

# Endpoint lấy thông tin repository cụ thể
@repo_router.get("/github/{owner}/{repo}")
async def fetch_repo(owner: str, repo: str):
    return await get_repo_data(owner, repo)

@repo_router.get("/github/repos")
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

# Save repo vào database
@repo_router.post("/github/{owner}/{repo}/save")
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
        return {"message": f"Repository {owner}/{repo} saved successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving repository: {str(e)}")