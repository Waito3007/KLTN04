from fastapi import APIRouter, Request, HTTPException, Query
import httpx
import logging
from typing import List, Dict, Any, Optional
from services.repo_service import get_all_repositories
from datetime import datetime, timezone
from utils.datetime_utils import normalize_github_datetime
import asyncio
from .sync_events import emit_sync_start, emit_sync_progress, emit_sync_complete, emit_sync_error

repo_manager_router = APIRouter()
logger = logging.getLogger(__name__)

async def github_api_call(url: str, token: str) -> Dict[str, Any]:
    """Make GitHub API call with proper error handling and connection retry"""
    headers = {"Authorization": token, "Accept": "application/vnd.github.v3+json"}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            timeout = httpx.Timeout(30.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    return None
                else:
                    logger.warning(f"GitHub API error {response.status_code}: {response.text}")
                    return None
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
            logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries} for {url}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"GitHub API call failed after {max_retries} attempts for {url}: {e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error in GitHub API call for {url}: {e}")
            return None

@repo_manager_router.get("/github/user/repositories")
async def get_user_repositories(request: Request, page: int = Query(1, ge=1), per_page: int = Query(30, le=100)):
    """
    Lấy danh sách repositories của user từ GitHub API
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    try:
        # Get user info first
        user_data = await github_api_call("https://api.github.com/user", token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid GitHub token")
        
        username = user_data["login"]
        
        # Get user repositories
        repos_url = f"https://api.github.com/user/repos?page={page}&per_page={per_page}&sort=updated&direction=desc"
        repos_data = await github_api_call(repos_url, token)
        
        if repos_data is None:
            raise HTTPException(status_code=500, detail="Failed to fetch repositories")
        
        # Format repository data
        formatted_repos = []
        for repo in repos_data:
            formatted_repo = {
                "github_id": repo["id"],
                "name": repo["name"],
                "full_name": repo["full_name"],
                "owner": repo["owner"]["login"],
                "description": repo.get("description", ""),
                "url": repo["html_url"],
                "clone_url": repo["clone_url"],
                "is_private": repo["private"],
                "is_fork": repo["fork"],
                "default_branch": repo.get("default_branch", "main"),
                "language": repo.get("language"),
                "stars": repo["stargazers_count"],
                "forks": repo["forks_count"],
                "open_issues": repo["open_issues_count"],
                "created_at": repo["created_at"],
                "updated_at": repo["updated_at"],
                "pushed_at": repo.get("pushed_at"),
                "size": repo["size"],
                "archived": repo.get("archived", False),
                "disabled": repo.get("disabled", False)
            }
            formatted_repos.append(formatted_repo)
        
        return {
            "repositories": formatted_repos,
            "user": {
                "login": username,
                "name": user_data.get("name"),
                "avatar_url": user_data.get("avatar_url")
            },
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": len(formatted_repos)
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching user repositories: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching repositories: {str(e)}")

@repo_manager_router.get("/repositories/sync-status")
async def get_repositories_sync_status(request: Request):
    """
    Lấy danh sách repositories đã đồng bộ trong database và so sánh với GitHub
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    try:
        # Get repositories from database
        db_repositories = await get_all_repositories()
        
        # Get user's GitHub repositories for comparison
        user_data = await github_api_call("https://api.github.com/user", token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid GitHub token")
        
        # Get all user repos from GitHub (multiple pages if needed)
        all_github_repos = []
        page = 1
        per_page = 100
        
        while True:
            repos_url = f"https://api.github.com/user/repos?page={page}&per_page={per_page}&sort=updated&direction=desc"
            repos_data = await github_api_call(repos_url, token)
            
            if not repos_data or len(repos_data) == 0:
                break
            
            all_github_repos.extend(repos_data)
            
            if len(repos_data) < per_page:
                break
            
            page += 1
        
        # Create lookup dictionaries
        db_repos_by_github_id = {repo["github_id"]: repo for repo in db_repositories}
        github_repos_by_id = {repo["id"]: repo for repo in all_github_repos}
        
        # Categorize repositories
        synced_repos = []
        unsynced_repos = []
        outdated_repos = []
        
        # Check GitHub repos against database
        for github_repo in all_github_repos:
            github_id = github_repo["id"]
            
            if github_id in db_repos_by_github_id:
                db_repo = db_repos_by_github_id[github_id]
                
                # Check if repo is outdated (GitHub updated_at > DB last_synced)
                github_updated = normalize_github_datetime(github_repo["updated_at"])
                db_last_synced = db_repo.get("last_synced")
                
                is_outdated = False
                if db_last_synced:
                    # Normalize DB datetime to timezone-naive UTC
                    if isinstance(db_last_synced, str):
                        db_last_synced = normalize_github_datetime(db_last_synced)
                    elif hasattr(db_last_synced, 'replace') and db_last_synced.tzinfo is not None:
                        # Convert timezone-aware to timezone-naive UTC
                        db_last_synced = db_last_synced.astimezone(timezone.utc).replace(tzinfo=None)
                    
                    # Add tolerance - consider 5 minutes as acceptable sync lag
                    from datetime import timedelta
                    tolerance = timedelta(minutes=5)
                    is_outdated = github_updated > (db_last_synced + tolerance)
                else:
                    is_outdated = True
                
                repo_info = {
                    **db_repo,
                    "github_updated_at": github_repo["updated_at"],
                    "github_pushed_at": github_repo.get("pushed_at"),
                    "is_outdated": is_outdated,
                    "needs_initial_sync": False,  # Already in DB, so not needing initial sync
                    "sync_priority": "high" if is_outdated else "normal"
                }
                
                if is_outdated:
                    outdated_repos.append(repo_info)
                else:
                    synced_repos.append(repo_info)
            else:
                # Repository not in database - needs initial sync
                repo_info = {
                    "github_id": github_repo["id"],
                    "name": github_repo["name"],
                    "full_name": github_repo["full_name"],
                    "owner": github_repo["owner"]["login"],
                    "description": github_repo.get("description", ""),
                    "url": github_repo["html_url"],
                    "language": github_repo.get("language"),
                    "stars": github_repo["stargazers_count"],
                    "forks": github_repo["forks_count"],
                    "is_private": github_repo["private"],
                    "github_updated_at": github_repo["updated_at"],
                    "github_pushed_at": github_repo.get("pushed_at"),
                    "sync_status": "not_synced",
                    "sync_priority": "highest",
                    "is_outdated": False,
                    "needs_initial_sync": True
                }
                unsynced_repos.append(repo_info)
        
        # Sort by priority
        unsynced_repos.sort(key=lambda x: x["github_updated_at"], reverse=True)
        outdated_repos.sort(key=lambda x: x["github_updated_at"], reverse=True)
        synced_repos.sort(key=lambda x: x.get("last_synced", ""), reverse=True)
        
        return {
            "summary": {
                "total_github_repos": len(all_github_repos),
                "total_db_repos": len(db_repositories),
                "unsynced_count": len(unsynced_repos),
                "outdated_count": len(outdated_repos),
                "synced_count": len(synced_repos)
            },
            "repositories": {
                "unsynced": unsynced_repos,
                "outdated": outdated_repos,
                "synced": synced_repos
            },
            "user": {
                "login": user_data["login"],
                "name": user_data.get("name"),
                "avatar_url": user_data.get("avatar_url")
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting repositories sync status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting sync status: {str(e)}")

@repo_manager_router.post("/repositories/{owner}/{repo}/sync")
async def sync_single_repository(owner: str, repo: str, request: Request, sync_type: str = Query("basic", regex="^(basic|enhanced|optimized)$")):
    """
    Đồng bộ một repository cụ thể với các loại sync khác nhau
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    repo_key = f"{owner}/{repo}"
    
    try:
        # Emit sync start event
        await emit_sync_start(repo_key, sync_type)
        
        if sync_type == "basic":
            # Import here to avoid circular imports
            from api.routes.sync import sync_repository_basic
            result = await sync_repository_basic(owner, repo, request)
            
            # Update repository sync status after completion
            from services.repo_service import update_repo_sync_status
            await update_repo_sync_status(owner, repo, "completed")
            
            await emit_sync_complete(repo_key, True, {"type": "basic", "result": result})
            return result
        elif sync_type == "enhanced":
            from api.routes.sync import sync_repository_enhanced
            result = await sync_repository_enhanced(owner, repo, request)
            
            # Update repository sync status after completion
            from services.repo_service import update_repo_sync_status
            await update_repo_sync_status(owner, repo, "completed")
            
            await emit_sync_complete(repo_key, True, {"type": "enhanced", "result": result})
            return result
        elif sync_type == "optimized":
            from api.routes.sync import sync_all_optimized
            from fastapi import BackgroundTasks
            background_tasks = BackgroundTasks()
            result = await sync_all_optimized(owner, repo, request, background_tasks)
            
            # Note: optimized sync updates status in its own background task
            await emit_sync_complete(repo_key, True, {"type": "optimized", "result": result})
            return result
        else:
            raise HTTPException(status_code=400, detail="Invalid sync type")
            
    except Exception as e:
        logger.error(f"Error syncing repository {owner}/{repo}: {e}")
        await emit_sync_error(repo_key, str(e), sync_type)
        raise HTTPException(status_code=500, detail=f"Error syncing repository: {str(e)}")
