# backend/api/routes/contributors.py
"""
Contributors/Collaborators API Routes

Mục đích: Quản lý thông tin contributors và collaborators của repositories
- Sync collaborators từ GitHub API vào database
- Lưu thông tin users và mối quan hệ repository-collaborator
- Cung cấp API lấy danh sách collaborators cho frontend
"""

from fastapi import APIRouter, Request, HTTPException
import httpx
import asyncio
import logging
from typing import Dict, Any, Optional, List
from services.repo_service import get_repo_id_by_owner_and_name
from services.user_service import save_user
from services.collaborator_service import get_collaborators_with_user_info

contributors_router = APIRouter()
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

@contributors_router.post("/github/{owner}/{repo}/sync-collaborators")
async def sync_repository_collaborators(owner: str, repo: str, request: Request):
    """
    🎯 **MỤC ĐÍCH**: Sync collaborators for specific repository when user selects it
    
    **LƯU GÌ**:
    1. **Users table**: 
       - github_id, github_username, email, display_name, full_name
       - avatar_url, bio, location, company, blog, twitter_username
       - github_profile_url, github_created_at, is_active, is_verified
    
    2. **Repository_collaborators table**:
       - repository_id, user_id (FK to users.id), role (ADMIN/MAINTAIN/PUSH/TRIAGE/PULL)
       - permissions (JSON string), is_owner, commits_count, issues_count, prs_count
    
    **FLOW**:
    1. Get collaborators from GitHub API
    2. Get detailed user info for each collaborator
    3. Save users to users table (upsert)
    4. Map GitHub permissions to role enum
    5. Save collaborator relationships to repository_collaborators table
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    try:
        # Get repository ID
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository not found. Please sync repository first.")

        # Get collaborators from GitHub
        collaborators_data = await github_api_call(f"{GITHUB_API_BASE}/repos/{owner}/{repo}/collaborators", token)
        
        # Enhanced collaborator data
        enhanced_collaborators = []
        users_to_save = []
        
        for collab in collaborators_data:
            try:
                # Get detailed user info
                user_data = await github_api_call(f"{GITHUB_API_BASE}/users/{collab['login']}", token)
                
                # Prepare user entry for users table
                user_entry = {
                    "github_id": user_data["id"],
                    "github_username": user_data["login"],
                    "email": user_data.get("email"),
                    "display_name": user_data.get("name"),
                    "full_name": user_data.get("name"),
                    "avatar_url": user_data.get("avatar_url"),
                    "bio": user_data.get("bio"),
                    "location": user_data.get("location"),
                    "company": user_data.get("company"),
                    "blog": user_data.get("blog"),
                    "twitter_username": user_data.get("twitter_username"),
                    "github_profile_url": user_data.get("html_url"),
                    "github_created_at": user_data.get("created_at"),
                    "is_active": True,
                    "is_verified": False
                }
                users_to_save.append(user_entry)

                # Prepare repository_collaborator entry
                permissions = collab.get("permissions", {})
                
                # Map GitHub permissions to enum values
                if permissions.get("admin", False):
                    role = "ADMIN"
                elif permissions.get("maintain", False):
                    role = "MAINTAIN"
                elif permissions.get("push", False):
                    role = "PUSH"
                elif permissions.get("triage", False):
                    role = "TRIAGE"
                else:
                    role = "PULL"  # Default role
                
                collab_entry = {
                    "repository_id": repo_id,
                    "github_id": user_data["id"],  # Store for lookup
                    "github_username": user_data["login"],  # Store for lookup
                    "role": role,
                    "permissions": str(permissions),  # Store as JSON string
                    "is_owner": collab["login"] == owner,
                    "commits_count": 0,  # Will be updated by commit sync
                    "issues_count": 0,   # Will be updated by issue sync
                    "prs_count": 0       # Will be updated by PR sync
                }
                enhanced_collaborators.append(collab_entry)
                
            except Exception as e:
                logger.warning(f"Error processing collaborator {collab.get('login', 'unknown')}: {e}")

        # Save users and collaborators to database
        saved_users_count = await save_multiple_users(users_to_save)
        saved_collab_count = await save_multiple_repository_collaborators(enhanced_collaborators)
        
        return {
            "status": "success",
            "repository": f"{owner}/{repo}",
            "collaborators": enhanced_collaborators,
            "saved_users_count": saved_users_count,
            "saved_collaborators_count": saved_collab_count,
            "message": f"Successfully synced {saved_collab_count} collaborators and {saved_users_count} users"
        }
        
    except Exception as e:
        logger.error(f"Error syncing collaborators for {owner}/{repo}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to sync collaborators: {str(e)}")

@contributors_router.get("/github/{owner}/{repo}/contributors")
async def get_repository_contributors(owner: str, repo: str, request: Request):
    """
    🎯 **MỤC ĐÍCH**: Lấy danh sách contributors từ GitHub API (public contributors)
    
    **KHÁC BIỆT VỚI COLLABORATORS**:
    - Contributors: Những người đã contribute code (có commits)
    - Collaborators: Những người có quyền access repository
    
    **TRẢ VỀ**: Danh sách contributors với số lượng contributions
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    try:
        contributors_data = await github_api_call(f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contributors", token)
        
        # Format contributors data
        formatted_contributors = []
        for contrib in contributors_data:
            formatted_contributors.append({
                "login": contrib["login"],
                "avatar_url": contrib["avatar_url"],
                "contributions": contrib["contributions"],
                "type": "Contributor",
                "github_url": contrib.get("html_url")
            })
        
        return {
            "repository": f"{owner}/{repo}",
            "contributors": formatted_contributors,
            "total_contributors": len(formatted_contributors)
        }
        
    except Exception as e:
        logger.error(f"Error fetching contributors for {owner}/{repo}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch contributors: {str(e)}")

@contributors_router.get("/repos/{owner}/{repo}/collaborators")
async def get_repository_collaborators_from_db(owner: str, repo: str):
    """
    🎯 **MỤC ĐÍCH**: Lấy danh sách collaborators từ database (đã sync)
    
    **LẤY TỪ DATABASE**:
    - repository_collaborators table JOIN users table
    - Bao gồm thông tin role, permissions, user details
    
    **SỬ DỤNG**: Frontend task assignment, filters
    """
    try:
        collaborators = await get_collaborators_with_user_info(owner, repo)
        
        return {
            "repository": f"{owner}/{repo}",
            "collaborators": collaborators,
            "count": len(collaborators)
        }
        
    except Exception as e:
        logger.error(f"Error fetching collaborators from DB for {owner}/{repo}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching collaborators: {str(e)}")

@contributors_router.get("/repos/{owner}/{repo}/collaborators/stats")
async def get_collaborators_statistics(owner: str, repo: str):
    """
    🎯 **MỤC ĐÍCH**: Thống kê collaborators theo role, activity
    
    **THỐNG KÊ**:
    - Số lượng theo role (ADMIN, PUSH, PULL, etc.)
    - Most active collaborators
    - Recent activity
    """
    try:
        from services.collaborator_service import get_collaborator_statistics
        
        stats = await get_collaborator_statistics(owner, repo)
        
        return {
            "repository": f"{owner}/{repo}",
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error fetching collaborator stats for {owner}/{repo}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching collaborator statistics: {str(e)}")

@contributors_router.get("/test")
async def test_contributors_router():
    """Test endpoint to verify contributors router is working"""
    return {"message": "Contributors router is working", "status": "ok"}

# ==================== HELPER FUNCTIONS ====================

async def save_multiple_users(users_list: List[dict]) -> int:
    """
    🎯 **MỤC ĐÍCH**: Lưu nhiều users cùng lúc với upsert logic
    
    **LƯU VÀO**: users table
    **LOGIC**: Nếu github_id đã tồn tại thì update, nếu không thì insert
    """
    if not users_list:
        logger.info("No users to save")
        return 0
    
    logger.info(f"Attempting to save {len(users_list)} users")
    
    try:
        saved_count = 0
        for user in users_list:
            try:
                logger.info(f"Saving user: {user.get('github_username', 'unknown')} (github_id: {user.get('github_id', 'unknown')})")
                await save_user(user)
                saved_count += 1
                logger.info(f"Successfully saved user: {user.get('github_username')}")
            except Exception as e:
                logger.warning(f"Error saving user {user.get('github_username', 'unknown')}: {e}")
        
        logger.info(f"Successfully saved {saved_count} out of {len(users_list)} users")
        return saved_count
        
    except Exception as e:
        logger.error(f"Error saving users: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 0

async def save_multiple_repository_collaborators(collaborators_list: List[dict]) -> int:
    """
    🎯 **MỤC ĐÍCH**: Lưu quan hệ repository-collaborator vào database
    
    **LƯU VÀO**: repository_collaborators table
    **LOGIC**: 
    1. Xóa tất cả collaborators cũ của repo
    2. Lookup user_id từ github_id
    3. Insert collaborator relationships mới
    """
    if not collaborators_list:
        logger.info("No collaborators to save")
        return 0
    
    logger.info(f"Attempting to save {len(collaborators_list)} collaborators")
    
    try:
        from db.database import database
        from db.models.repository_collaborators import repository_collaborators
        from db.models.users import users
        from sqlalchemy import insert, delete, select
        
        # Clear existing collaborators for this repository
        if collaborators_list:
            repo_id = collaborators_list[0]["repository_id"]
            logger.info(f"Clearing existing collaborators for repo_id: {repo_id}")
            delete_query = delete(repository_collaborators).where(
                repository_collaborators.c.repository_id == repo_id
            )
            await database.execute(delete_query)
            logger.info(f"Deleted existing collaborators for repo {repo_id}")
        
        # Process collaborators to get proper user_id
        processed_collaborators = []
        for collab in collaborators_list:
            logger.info(f"Processing collaborator: {collab['github_username']} (github_id: {collab['github_id']})")
            
            # Lookup user_id from github_id
            user_lookup = select(users.c.id).where(
                users.c.github_id == collab["github_id"]
            )
            user_result = await database.fetch_one(user_lookup)
            
            if user_result:
                logger.info(f"Found user_id {user_result.id} for github_id {collab['github_id']}")
                processed_collab = {
                    "repository_id": collab["repository_id"],
                    "user_id": user_result.id,  # Use internal user_id
                    "role": collab["role"],
                    "permissions": collab["permissions"],
                    "is_owner": collab["is_owner"],
                    "commits_count": collab.get("commits_count", 0),
                    "issues_count": collab.get("issues_count", 0),
                    "prs_count": collab.get("prs_count", 0)
                }
                processed_collaborators.append(processed_collab)
                logger.info(f"Processed collaborator: {processed_collab}")
            else:
                logger.warning(f"User not found for github_id: {collab['github_id']} (username: {collab['github_username']})")
        
        logger.info(f"About to insert {len(processed_collaborators)} processed collaborators")
        
        # Insert new collaborators
        if processed_collaborators:
            insert_query = insert(repository_collaborators)
            for collab in processed_collaborators:
                await database.execute(insert_query, collab)
            logger.info(f"Inserted {len(processed_collaborators)} collaborators")
        else:
            logger.warning("No processed collaborators to insert")
        
        logger.info(f"Successfully saved {len(processed_collaborators)} collaborators")
        return len(processed_collaborators)
        
    except Exception as e:
        logger.error(f"Error saving repository collaborators: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 0
