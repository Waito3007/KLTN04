# backend/api/routes/repo.py
from fastapi import APIRouter, Request, HTTPException, Query
import httpx
from typing import Optional, List
from services.repo_service import (
    save_repository, fetch_repo_from_github, fetch_repo_from_database,
    get_user_repos_from_database, get_repositories_by_owner, get_repository_stats,
    get_repo_id_by_owner_and_name
)
from services.collaborator_service import (
    get_collaborators_by_repo, sync_repository_collaborators
)

repo_router = APIRouter()

# Endpoint lấy thông tin repository cụ thể từ GitHub
@repo_router.get("/github/{owner}/{repo}")
async def fetch_repo(owner: str, repo: str):
    return await fetch_repo_from_github(owner, repo)

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

# Endpoint lấy thông tin repository từ database
@repo_router.get("/repodb/{owner}/{repo}")
async def get_repo_from_database(owner: str, repo: str):
    """Fetch repository information from database"""
    try:
        repo_data = await fetch_repo_from_database(owner, repo)
        if not repo_data:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo} not found in database")
        return repo_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching repository from database: {str(e)}")

#Endpoint lấy danh sách repositories từ database
@repo_router.get("/repodb/repos")
async def get_repos_from_database(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    owner: Optional[str] = Query(None, description="Filter by owner"),
    limit: Optional[int] = Query(50, description="Limit number of results"),
    offset: Optional[int] = Query(0, description="Offset for pagination")
):
    """Fetch repositories from database with optional filtering"""
    try:
        if owner:
            # Lấy repositories theo owner
            repos = await get_repositories_by_owner(owner, limit, offset)
        elif user_id:
            # Lấy repositories theo user_id
            repos = await get_user_repos_from_database(user_id, limit, offset)
        else:
            # Lấy tất cả repositories
            repos = await get_user_repos_from_database(None, limit, offset)
        
        return {
            "repositories": repos,
            "count": len(repos),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching repositories from database: {str(e)}")

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

# Endpoint lấy collaborators từ database
@repo_router.get("/github/{owner}/{repo}/collaborators")
async def get_repo_collaborators(owner: str, repo: str):
    """Fetch collaborators for a repository from database"""
    try:
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo} not found in database")
        
        collaborators = await get_collaborators_by_repo(repo_id)
        
        return {
            "repository": f"{owner}/{repo}",
            "collaborators": collaborators,
            "count": len(collaborators)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching collaborators: {str(e)}")

# Sync collaborators từ GitHub API và lấy danh sách cập nhật từ database
@repo_router.post("/github/{owner}/{repo}/collaborators/sync")
async def sync_and_get_collaborators(owner: str, repo: str, request: Request):
    """
    Sync collaborators from GitHub API and fetch updated list from database
    
    Args:
        owner: Repository owner
        repo: Repository name
        request: HTTP request object

    Returns:
        JSON response with updated collaborator list
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    try:
        # Fetch repository ID from database
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo} not found in database")

        # Sync collaborators from GitHub API
        sync_result = await sync_repository_collaborators(owner, repo, token.split(" ")[1])

        # Fetch updated collaborators from database
        collaborators = await get_collaborators_by_repo(repo_id)

        return {
            "repository": f"{owner}/{repo}",
            "collaborators": collaborators,
            "sync_result": sync_result,
            "count": len(collaborators)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing and fetching collaborators: {str(e)}")