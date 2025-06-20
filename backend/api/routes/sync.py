# backend/api/routes/sync.py
from fastapi import APIRouter, Request, HTTPException
import httpx
import asyncio
import logging
from typing import Dict, Any, Optional
from services.repo_service import save_repository, get_repo_id_by_owner_and_name
from services.branch_service import sync_branches_for_repo
from services.commit_service import save_commit
from services.issue_service import save_issue
from services.github_service import fetch_commit_details, fetch_branch_stats

sync_router = APIRouter()
logger = logging.getLogger(__name__)

# Constants
GITHUB_API_BASE = "https://api.github.com"

async def github_api_call(url: str, token: str, retries: int = 3) -> Dict[str, Any]:
    """
    Gọi GitHub API với error handling và retry logic
    
    Args:
        url: GitHub API URL
        token: Authorization token
        retries: Số lần retry nếu rate limit
    
    Returns:
        Response JSON data
    
    Raises:
        HTTPException: Khi API call thất bại
    """
    headers = {
        "Authorization": token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    for attempt in range(retries + 1):
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(url, headers=headers)
                
                # Handle rate limiting
                if resp.status_code == 429:
                    if attempt < retries:
                        reset_time = int(resp.headers.get("X-RateLimit-Reset", "0"))
                        wait_time = min(reset_time - int(asyncio.get_event_loop().time()), 60)
                        logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}")
                        await asyncio.sleep(max(wait_time, 1))
                        continue
                    else:
                        raise HTTPException(
                            status_code=429, 
                            detail="GitHub API rate limit exceeded. Please try again later."
                        )
                
                # Handle other HTTP errors
                if resp.status_code != 200:
                    error_detail = f"GitHub API error: {resp.status_code}"
                    try:
                        error_data = resp.json()
                        error_detail += f" - {error_data.get('message', resp.text)}"
                    except:
                        error_detail += f" - {resp.text}"
                    
                    raise HTTPException(status_code=resp.status_code, detail=error_detail)
                
                return resp.json()
                
            except httpx.TimeoutException:
                if attempt < retries:
                    logger.warning(f"Request timeout, retrying... (attempt {attempt + 1})")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise HTTPException(status_code=408, detail="GitHub API request timeout")
            
            except httpx.RequestError as e:
                if attempt < retries:
                    logger.warning(f"Request error: {e}, retrying... (attempt {attempt + 1})")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise HTTPException(status_code=500, detail=f"GitHub API request failed: {str(e)}")
    
    raise HTTPException(status_code=500, detail="All retry attempts failed")

# Đồng bộ toàn bộ dữ liệu
@sync_router.post("/github/{owner}/{repo}/sync-all")
async def sync_all(owner: str, repo: str, request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    try:
        # 1. Sync repository
        repo_data = await github_api_call(f"https://api.github.com/repos/{owner}/{repo}", token)
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
            # Bổ sung các fields từ database model
            "full_name": repo_data.get("full_name"),
            "clone_url": repo_data.get("clone_url"),
            "is_private": repo_data.get("private", False),
            "is_fork": repo_data.get("fork", False),
            "default_branch": repo_data.get("default_branch", "main"),
            "sync_status": "completed",        }
        await save_repository(repo_entry)
        
        # 2. Sync branches
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository not found")
            
        branches_data = await github_api_call(f"https://api.github.com/repos/{owner}/{repo}/branches", token)
          # Chuẩn hóa dữ liệu branch với đầy đủ thông tin
        default_branch = repo_data.get("default_branch", "main")
        branches_to_save = []
        
        for branch in branches_data:
            branch_info = {
                "name": branch["name"],
                "sha": branch.get("commit", {}).get("sha"),
                "is_default": branch["name"] == default_branch,
                "is_protected": branch.get("protected", False),
            }
            
            # Tùy chọn: Lấy thêm thông tin commit chi tiết (có thể làm chậm API)
            # Uncomment dòng dưới nếu muốn lấy thêm thông tin
            # if branch_info["sha"]:
            #     commit_details = await fetch_commit_details(branch_info["sha"], owner, repo, token)
            #     if commit_details:
            #         branch_info["last_commit_date"] = commit_details["date"]
            #         branch_info["last_committer_name"] = commit_details["committer_name"]
            
            branches_to_save.append(branch_info)
        
        # Đồng bộ hóa hàng loạt với dữ liệu đầy đủ
        branches_synced = await sync_branches_for_repo(
            repo_id, 
            branches_to_save, 
            default_branch=default_branch,
            replace_existing=True
        )
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
        repo_data = await github_api_call(f"https://api.github.com/repos/{owner}/{repo}", token)
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
            "full_name": repo_data.get("full_name"),
            "clone_url": repo_data.get("clone_url"),
            "is_private": repo_data.get("private", False),
            "is_fork": repo_data.get("fork", False),
            "default_branch": repo_data.get("default_branch", "main"),
            "sync_status": "completed",
        }
        await save_repository(repo_entry)
        
        return {"message": f"Đồng bộ cơ bản {owner}/{repo} thành công!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đồng bộ cơ bản {owner}/{repo}: {str(e)}")

# Endpoint đồng bộ nâng cao - bao gồm thông tin commit chi tiết
@sync_router.post("/github/{owner}/{repo}/sync-enhanced")
async def sync_enhanced(owner: str, repo: str, request: Request):
    """
    Đồng bộ repository với thông tin chi tiết bao gồm:
    - Thông tin repository đầy đủ
    - Thông tin branch đầy đủ
    - Thông tin commit cuối cùng cho mỗi branch
    - Thống kê branch (nếu có)
    
    Lưu ý: Endpoint này sẽ chậm hơn do phải gọi nhiều API calls
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    try:
        logger.info(f"Starting enhanced sync for {owner}/{repo}")
        
        # 1. Sync repository
        repo_data = await github_api_call(f"https://api.github.com/repos/{owner}/{repo}", token)
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
            "full_name": repo_data.get("full_name"),
            "clone_url": repo_data.get("clone_url"),
            "is_private": repo_data.get("private", False),
            "is_fork": repo_data.get("fork", False),
            "default_branch": repo_data.get("default_branch", "main"),
            "sync_status": "enhanced_completed",
        }
        await save_repository(repo_entry)
        
        # 2. Sync branches với thông tin chi tiết
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        branches_data = await github_api_call(f"https://api.github.com/repos/{owner}/{repo}/branches", token)
        
        default_branch = repo_data.get("default_branch", "main")
        branches_to_save = []
        
        # Process branches with enhanced data
        for i, branch in enumerate(branches_data):
            logger.info(f"Processing branch {i+1}/{len(branches_data)}: {branch['name']}")
            
            branch_info = {
                "name": branch["name"],
                "sha": branch.get("commit", {}).get("sha"),
                "is_default": branch["name"] == default_branch,
                "is_protected": branch.get("protected", False),
            }
            
            # Lấy thông tin commit chi tiết cho branch
            if branch_info["sha"]:
                try:
                    commit_details = await fetch_commit_details(
                        branch_info["sha"], owner, repo, token
                    )
                    if commit_details:
                        branch_info.update({
                            "last_commit_date": commit_details.get("date"),
                            "last_committer_name": commit_details.get("committer_name"),
                            "creator_name": commit_details.get("author_name"),  # Assuming first commit author as creator
                        })
                except Exception as e:
                    logger.warning(f"Failed to fetch commit details for branch {branch['name']}: {e}")
            
            # Lấy thống kê branch (optional)
            try:
                branch_stats = await fetch_branch_stats(owner, repo, branch["name"], token)
                if branch_stats:
                    branch_info.update({
                        "commits_count": branch_stats.get("commits_count"),
                        "contributors_count": branch_stats.get("contributors_count"),
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch branch stats for {branch['name']}: {e}")
            
            branches_to_save.append(branch_info)
            
            # Add small delay to avoid hitting rate limits too hard
            if i < len(branches_data) - 1:  # Don't sleep after last branch
                await asyncio.sleep(0.1)
        
        # Đồng bộ hóa hàng loạt với dữ liệu đầy đủ
        branches_synced = await sync_branches_for_repo(
            repo_id, 
            branches_to_save, 
            default_branch=default_branch,
            replace_existing=True
        )
        
        logger.info(f"Enhanced sync completed for {owner}/{repo}: {branches_synced} branches synced")
        
        return {
            "message": f"Đồng bộ nâng cao {owner}/{repo} thành công!",
            "repository_synced": True,
            "branches_synced": branches_synced,
            "enhanced_data": True
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Enhanced sync error for {owner}/{repo}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi đồng bộ nâng cao {owner}/{repo}: {str(e)}")

# Endpoint kiểm tra trạng thái GitHub API và token
@sync_router.get("/github/status")
async def github_status(request: Request):
    """
    Kiểm tra trạng thái kết nối GitHub API và thông tin rate limit
    
    Returns:
        dict: Thông tin về token, rate limit, và trạng thái API
    """
    from services.github_service import validate_github_token, get_rate_limit_info
    
    token = request.headers.get("Authorization", "").replace("token ", "")
    
    result = {
        "github_api_accessible": False,
        "token_valid": False,
        "rate_limit": None,
        "token_provided": bool(token)
    }
    
    try:
        # Kiểm tra token nếu được cung cấp
        if token:
            result["token_valid"] = await validate_github_token(token)
        
        # Lấy thông tin rate limit
        rate_limit_info = await get_rate_limit_info(token if token else None)
        result["rate_limit"] = rate_limit_info.get("resources", {}).get("core", {})
        result["github_api_accessible"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

# Endpoint lấy danh sách repositories có sẵn cho user
@sync_router.get("/github/repositories")
async def list_user_repositories(request: Request, per_page: int = 30, page: int = 1):
    """
    Lấy danh sách repositories của user hiện tại
    
    Args:
        per_page: Số repo trên mỗi trang (max 100)
        page: Số trang
    
    Returns:
        list: Danh sách repositories
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    try:
        # Giới hạn per_page để tránh quá tải
        per_page = min(max(per_page, 1), 100)
        page = max(page, 1)
        
        url = f"https://api.github.com/user/repos?per_page={per_page}&page={page}&sort=updated"
        repos_data = await github_api_call(url, token)
        
        # Trả về thông tin cơ bản của các repos
        simplified_repos = []
        for repo in repos_data:
            simplified_repos.append({
                "id": repo["id"],
                "name": repo["name"],
                "full_name": repo["full_name"],
                "owner": repo["owner"]["login"],
                "description": repo.get("description"),
                "language": repo.get("language"),
                "stars": repo["stargazers_count"],
                "forks": repo["forks_count"],
                "updated_at": repo["updated_at"],
                "is_private": repo["private"],
                "is_fork": repo["fork"],
                "default_branch": repo.get("default_branch", "main")
            })
        
        return {
            "repositories": simplified_repos,
            "page": page,
            "per_page": per_page,
            "total_returned": len(simplified_repos)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lấy danh sách repositories: {str(e)}")

# Endpoint thống kê repository và branches
@sync_router.get("/github/{owner}/{repo}/stats")
async def get_repository_stats(owner: str, repo: str):
    """
    Lấy thống kê chi tiết về repository và branches
    
    Returns:
        dict: Thống kê repository và branches
    """
    from services.branch_service import get_branch_statistics, find_stale_branches, get_most_active_branches
    from services.repo_service import get_repository_by_owner_and_name
    
    try:
        # Lấy thông tin repository
        repo_info = await get_repository_by_owner_and_name(owner, repo)
        if not repo_info:
            raise HTTPException(status_code=404, detail="Repository not found in database")
        
        repo_id = repo_info['id']
        
        # Lấy thống kê branches
        branch_stats = await get_branch_statistics(repo_id)
        
        # Lấy branches cũ (90 ngày)
        stale_branches = await find_stale_branches(repo_id, days_threshold=90)
        
        # Lấy branches hoạt động nhất
        active_branches = await get_most_active_branches(repo_id, limit=5)
        
        return {
            "repository": {
                "name": repo_info['name'],
                "owner": repo_info['owner'],
                "stars": repo_info.get('stars', 0),
                "forks": repo_info.get('forks', 0),
                "language": repo_info.get('language'),
                "last_synced": repo_info.get('updated_at'),
                "sync_status": repo_info.get('sync_status', 'unknown')
            },
            "branch_statistics": branch_stats,
            "stale_branches": {
                "count": len(stale_branches),
                "branches": [{"name": b["name"], "last_commit_date": b["last_commit_date"]} for b in stale_branches[:10]]
            },
            "most_active_branches": [
                {
                    "name": b["name"], 
                    "commits_count": b["commits_count"],
                    "is_default": b["is_default"],
                    "is_protected": b["is_protected"]
                } 
                for b in active_branches
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lấy thống kê {owner}/{repo}: {str(e)}")

# Endpoint để cập nhật branch metadata
@sync_router.patch("/github/{owner}/{repo}/branches/{branch_name}")
async def update_branch_info(owner: str, repo: str, branch_name: str, metadata: dict, request: Request):
    """
    Cập nhật thông tin metadata của một branch
    
    Args:
        owner: Chủ sở hữu repository
        repo: Tên repository
        branch_name: Tên branch
        metadata: Dữ liệu cần cập nhật
    
    Returns:
        dict: Kết quả cập nhật
    """
    from services.branch_service import update_branch_metadata
    from services.repo_service import get_repo_id_by_owner_and_name
    
    # Optional: Kiểm tra token nếu cần authorization
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    try:
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        success = await update_branch_metadata(repo_id, branch_name, metadata)
        
        if success:
            return {
                "message": f"Branch {branch_name} updated successfully",
                "repository": f"{owner}/{repo}",
                "branch": branch_name,
                "updated_fields": list(metadata.keys())
            }
        else:
            raise HTTPException(status_code=404, detail="Branch not found or no valid fields to update")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi cập nhật branch {branch_name}: {str(e)}")

# ==================== AUTO-SYNC ENDPOINTS FOR REPOSITORY SELECTION ====================

@sync_router.post("/github/{owner}/{repo}/sync-branches")
async def sync_repository_branches(owner: str, repo: str, request: Request):
    """
    Sync branches for specific repository when user selects it
    Auto-creates repository if not exists
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    try:
        # Get repository ID, create if not exists
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            # Auto-create repository
            repo_data = await github_api_call(f"{GITHUB_API_BASE}/repos/{owner}/{repo}", token)
            repo_entry = {
                "github_id": repo_data["id"],
                "name": repo_data["name"],
                "owner": repo_data["owner"]["login"],
                "description": repo_data.get("description"),
                "full_name": repo_data.get("full_name"),
                "default_branch": repo_data.get("default_branch", "main"),
                "stars": repo_data.get("stargazers_count", 0),
                "forks": repo_data.get("forks_count", 0),
                "language": repo_data.get("language"),
                "is_private": repo_data.get("private", False),
                "sync_status": "auto_created"
            }
            await save_repository(repo_entry)
            repo_id = await get_repo_id_by_owner_and_name(owner, repo)

        # Sync branches
        branches_data = await github_api_call(f"{GITHUB_API_BASE}/repos/{owner}/{repo}/branches", token)
        
        # Enhanced branch data with commit info
        enhanced_branches = []
        default_branch = None
        
        # Get repository info for default branch
        try:
            repo_info = await github_api_call(f"{GITHUB_API_BASE}/repos/{owner}/{repo}", token)
            default_branch = repo_info.get("default_branch", "main")
        except:
            default_branch = "main"
        
        for branch in branches_data:
            try:
                # Get additional commit info for each branch
                commit_data = await github_api_call(
                    f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{branch['commit']['sha']}", 
                    token
                )
                
                enhanced_branch = {
                    "name": branch["name"],
                    "sha": branch["commit"]["sha"],
                    "is_protected": branch.get("protected", False),
                    "is_default": branch["name"] == default_branch,
                    "repo_id": repo_id,
                    "creator_user_id": None,  # Could enhance later
                    "last_committer_user_id": None,  # Could enhance later
                    "commits_count": 1,  # Basic count, could enhance
                    "contributors_count": 1,  # Basic count, could enhance
                    "last_commit_date": commit_data["commit"]["committer"]["date"]
                }
                enhanced_branches.append(enhanced_branch)
                
            except Exception as e:
                logger.warning(f"Error getting commit info for branch {branch['name']}: {e}")
                # Fallback to basic branch info
                enhanced_branches.append({
                    "name": branch["name"],
                    "sha": branch["commit"]["sha"],
                    "is_protected": branch.get("protected", False),
                    "is_default": branch["name"] == default_branch,
                    "repo_id": repo_id,
                    "commits_count": 1,
                    "contributors_count": 1
                })        # Save branches to database
        saved_count = await save_multiple_branches(repo_id, enhanced_branches)
        
        return {
            "status": "success",
            "repository": f"{owner}/{repo}",
            "branches": enhanced_branches,
            "saved_count": saved_count,
            "message": f"Successfully synced {saved_count} branches"
        }
        
    except Exception as e:
        logger.error(f"Error syncing branches for {owner}/{repo}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to sync branches: {str(e)}")

# ==================== HELPER FUNCTIONS ====================

async def save_multiple_branches(repo_id: int, branches_list: list):
    """Save multiple branches efficiently"""
    if not branches_list:
        return 0
    
    try:
        from db.models.branches import branches
        from db.database import engine
        from sqlalchemy import insert, delete
        
        with engine.connect() as conn:
            # Clear existing branches for this repository
            delete_query = delete(branches).where(branches.c.repo_id == repo_id)
            conn.execute(delete_query)
            
            # Insert new branches
            if branches_list:
                insert_query = insert(branches)
                conn.execute(insert_query, branches_list)
            
            conn.commit()
        
        return len(branches_list)
        
    except Exception as e:
        logger.error(f"Error saving branches: {e}")
        return 0
