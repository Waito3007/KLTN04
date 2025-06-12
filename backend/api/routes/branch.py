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

# Lưu branch vào database
@branch_router.post("/github/{owner}/{repo}/save-branches")
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

    saved_count = 0
    for branch in branches:
        try:
            branch_data = {
                "name": branch["name"],
                "repo_id": repo_id,
            }
            await save_branch(branch_data)
            saved_count += 1
        except Exception as e:
            print(f"Lỗi khi lưu branch {branch['name']}: {e}")
            continue

    return {"message": f"Đã lưu {saved_count}/{len(branches)} branches!"}
