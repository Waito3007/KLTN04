# backend/services/collaborator_service.py
from db.models.repository_collaborators import repository_collaborators
from db.models.users import users
from db.models.repositories import repositories
from db.models.collaborators import collaborators
from sqlalchemy import select, join, func, insert, update
from db.database import database
import logging
from typing import List, Dict, Any, Optional
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)

async def get_collaborators_by_repo(repo_id: int) -> List[Dict[str, Any]]:
    """
    Get all collaborators for a specific repository using the new schema
    
    Args:
        repo_id: Repository ID
        
    Returns:
        List of collaborator data with user information
    """
    try:
        # Join repository_collaborators with collaborators table to get full user info
        query = (
            select(
                repository_collaborators.c.id,
                repository_collaborators.c.repository_id,
                repository_collaborators.c.collaborator_id,
                repository_collaborators.c.role,
                repository_collaborators.c.permissions,
                repository_collaborators.c.is_owner,
                repository_collaborators.c.commits_count,
                repository_collaborators.c.issues_count,
                repository_collaborators.c.prs_count,
                repository_collaborators.c.joined_at,
                repository_collaborators.c.last_synced.label('collab_last_synced'),
                # Collaborator info from collaborators table
                collaborators.c.github_user_id,
                collaborators.c.github_username,
                collaborators.c.display_name,
                collaborators.c.email,
                collaborators.c.avatar_url,
                collaborators.c.bio,
                collaborators.c.company,
                collaborators.c.location,
                collaborators.c.blog,
                collaborators.c.is_site_admin,
                collaborators.c.type.label('collaborator_type'),
                collaborators.c.user_id  # Link to users table if they've logged in
            )
            .select_from(
                repository_collaborators.join(
                    collaborators,
                    repository_collaborators.c.collaborator_id == collaborators.c.id,
                    isouter=True
                )
            )
            .where(repository_collaborators.c.repository_id == repo_id)
            .order_by(repository_collaborators.c.is_owner.desc(), repository_collaborators.c.role.desc())
        )
        
        results = await database.fetch_all(query)
        
        collaborators_list = []
        for row in results:
            collab_data = {
                "id": row.id,
                "repository_id": row.repository_id,
                "collaborator_id": row.collaborator_id,
                "github_user_id": row.github_user_id,
                "github_username": row.github_username,
                "login": row.github_username,  # For frontend compatibility
                "role": row.role,
                "permissions": row.permissions,
                "is_owner": row.is_owner,
                "commits_count": row.commits_count or 0,
                "issues_count": row.issues_count or 0,
                "prs_count": row.prs_count or 0,
                "contributions": row.commits_count or 0,  # For frontend compatibility
                "type": "Owner" if row.is_owner else row.role.capitalize() if row.role else "Collaborator",
                "joined_at": row.joined_at,
                "last_synced": row.collab_last_synced,
                # Collaborator info
                "display_name": row.display_name,
                "email": row.email,
                "avatar_url": row.avatar_url,
                "bio": row.bio,
                "location": row.location,
                "company": row.company,
                "blog": row.blog,
                "is_site_admin": row.is_site_admin,                "collaborator_type": row.collaborator_type,
                "user_id": row.user_id  # If they have logged into our system
            }
            collaborators_list.append(collab_data)
        
        logger.info(f"✅ Found {len(collaborators_list)} synced collaborators for repo {repo_id}")
        return collaborators_list
          
    except Exception as e:
        logger.error(f"Error getting collaborators for repo {repo_id}: {e}")
        return []

async def get_collaborator_by_repo_and_username(repo_id: int, github_username: str):
    """
    Get specific collaborator by repository and GitHub username
    
    Args:
        repo_id: Repository ID
        github_username: GitHub username
        
    Returns:
        Collaborator data or None
    """
    try:
        query = (
            select(repository_collaborators)
            .where(
                (repository_collaborators.c.repository_id == repo_id) &
                (repository_collaborators.c.github_username == github_username)
            )
        )
        
        result = await database.fetch_one(query)
        return dict(result) if result else None
        
    except Exception as e:
        logger.error(f"Error getting collaborator {github_username} for repo {repo_id}: {e}")
        return None

async def update_collaborator_stats(repo_id: int, github_username: str, stats: Dict[str, int]):
    """
    Update collaborator statistics (commits, issues, PRs count)
    
    Args:
        repo_id: Repository ID
        github_username: GitHub username
        stats: Dictionary with commits_count, issues_count, prs_count
    """
    try:
        query = (
            update(repository_collaborators)
            .where(
                (repository_collaborators.c.repository_id == repo_id) &
                (repository_collaborators.c.github_username == github_username)
            )
            .values(
                commits_count=stats.get('commits_count', 0),
                issues_count=stats.get('issues_count', 0),
                prs_count=stats.get('prs_count', 0),
                updated_at=func.now()
            )
        )
        
        await database.execute(query)
        logger.info(f"Updated stats for collaborator {github_username} in repo {repo_id}")
        
    except Exception as e:
        logger.error(f"Error updating collaborator stats: {e}")
        raise e

async def get_collaborators_with_user_info(owner: str, repo: str) -> List[Dict[str, Any]]:
    """
    Get collaborators for a repository by owner/repo names with full user info
    Legacy function - now redirects to get_collaborators_with_fallback
    
    Args:
        owner: Repository owner (GitHub username)
        repo: Repository name
        
    Returns:
        List of collaborator data with user information
    """
    try:
        logger.info(f"🔍 Legacy function called for repository: {owner}/{repo}")
        
        # Redirect to new function
        return await get_collaborators_with_fallback(owner, repo, None)
        
    except Exception as e:
        logger.error(f"💥 Error in legacy function for {owner}/{repo}: {e}")
        return []

def create_fallback_collaborator(user_record, repo_id: int, is_owner: bool = False) -> Dict[str, Any]:
    """Create a fallback collaborator object from user record"""
    return {
        "id": 0,  # Special ID for fallback
        "repository_id": repo_id,
        "user_id": user_record.id,
        "github_id": user_record.github_id,
        "github_username": user_record.github_username,
        "login": user_record.github_username,
        "role": "ADMIN" if is_owner else "PUSH",
        "permissions": '{"admin": true, "push": true, "pull": true}' if is_owner else '{"push": true, "pull": true}',
        "is_owner": is_owner,
        "commits_count": 0,
        "issues_count": 0,
        "prs_count": 0,
        "contributions": 0,
        "type": "Owner (Auto-detected)" if is_owner else "User (Auto-detected)",
        "joined_at": None,
        "last_synced": None,
        "sync_status": "fallback",
        # User info
        "display_name": user_record.display_name,
        "full_name": user_record.full_name,
        "email": user_record.email,
        "avatar_url": user_record.avatar_url,
        "bio": user_record.bio,
        "location": user_record.location,
        "company": user_record.company,
        "blog": user_record.blog,
        "twitter_username": user_record.twitter_username,
        "github_profile_url": user_record.github_profile_url,
        "github_created_at": user_record.github_created_at
    }

async def upsert_collaborator(github_user_data: Dict[str, Any]) -> int:
    """
    Insert or update a collaborator in the collaborators table
    
    Args:
        github_user_data: GitHub user data from API
        
    Returns:
        collaborator_id: ID of the collaborator in our database
    """
    try:
        github_user_id = github_user_data.get("id")
        github_username = github_user_data.get("login")
        
        if not github_user_id or not github_username:
            raise ValueError("Missing required GitHub user data")
        
        # Check if collaborator already exists
        existing_query = select(collaborators.c.id).where(
            collaborators.c.github_user_id == github_user_id
        )
        existing = await database.fetch_one(existing_query)
        
        collaborator_data = {
            "github_user_id": github_user_id,
            "github_username": github_username,
            "display_name": github_user_data.get("name"),
            "email": github_user_data.get("email"),
            "avatar_url": github_user_data.get("avatar_url"),
            "bio": github_user_data.get("bio"),
            "company": github_user_data.get("company"),
            "location": github_user_data.get("location"),
            "blog": github_user_data.get("blog"),
            "is_site_admin": github_user_data.get("site_admin", False),
            "node_id": github_user_data.get("node_id"),
            "gravatar_id": github_user_data.get("gravatar_id"),
            "type": github_user_data.get("type", "User"),
            "updated_at": func.now()
        }
        
        if existing:
            # Update existing collaborator
            update_query = (
                update(collaborators)
                .where(collaborators.c.github_user_id == github_user_id)
                .values(**collaborator_data)
            )
            await database.execute(update_query)
            collaborator_id = existing.id
            logger.info(f"Updated collaborator {github_username} (ID: {collaborator_id})")
        else:
            # Insert new collaborator
            insert_query = insert(collaborators).values(**collaborator_data)
            collaborator_id = await database.execute(insert_query)
            logger.info(f"Created new collaborator {github_username} (ID: {collaborator_id})")
        
        return collaborator_id
        
    except Exception as e:
        logger.error(f"Error upserting collaborator {github_user_data.get('login', 'unknown')}: {e}")
        raise e

async def link_collaborator_to_repository(
    repository_id: int, 
    collaborator_id: int, 
    github_collab_data: Dict[str, Any]
) -> bool:
    """
    Link a collaborator to a repository with permissions and role
    
    Args:
        repository_id: Repository ID
        collaborator_id: Collaborator ID
        github_collab_data: GitHub collaborator data with permissions
        
    Returns:
        bool: True if successful
    """
    try:
        # Check if link already exists
        existing_query = select(repository_collaborators.c.id).where(
            (repository_collaborators.c.repository_id == repository_id) &
            (repository_collaborators.c.collaborator_id == collaborator_id)
        )
        existing = await database.fetch_one(existing_query)
        
        # Extract permissions and role from GitHub data
        permissions = github_collab_data.get("permissions", {})
        role = github_collab_data.get("role_name", "pull")  # Default to pull
        
        # Determine if this is the owner
        is_owner = (
            permissions.get("admin", False) and 
            github_collab_data.get("login") == github_collab_data.get("repository_owner")
        )
        
        link_data = {
            "repository_id": repository_id,
            "collaborator_id": collaborator_id,
            "role": role,
            "permissions": str(permissions),  # Store as JSON string
            "is_owner": is_owner,
            "joined_at": datetime.now(),
            "last_synced": func.now(),
            "updated_at": func.now()
        }
        
        if existing:
            # Update existing link
            update_query = (
                update(repository_collaborators)
                .where(repository_collaborators.c.id == existing.id)
                .values(**link_data)
            )
            await database.execute(update_query)
            logger.info(f"Updated repository-collaborator link (repo: {repository_id}, collaborator: {collaborator_id})")
        else:
            # Create new link
            insert_query = insert(repository_collaborators).values(**link_data)
            await database.execute(insert_query)
            logger.info(f"Created repository-collaborator link (repo: {repository_id}, collaborator: {collaborator_id})")
        
        return True
        
    except Exception as e:
        logger.error(f"Error linking collaborator {collaborator_id} to repository {repository_id}: {e}")
        raise e

async def sync_repository_collaborators(owner: str, repo: str, github_token: str) -> Dict[str, Any]:
    """
    Sync collaborators from GitHub API to our database
    
    Args:
        owner: Repository owner
        repo: Repository name  
        github_token: GitHub API token
        
    Returns:
        Sync result with counts and status
    """
    try:
        logger.info(f"🔄 Starting collaborator sync for {owner}/{repo}")
        
        # Get repository ID
        repo_query = select(repositories.c.id).where(
            (repositories.c.owner == owner) &
            (repositories.c.name == repo)
        )
        repo_result = await database.fetch_one(repo_query)
        
        if not repo_result:
            raise ValueError(f"Repository {owner}/{repo} not found in database")
        
        repository_id = repo_result.id
        
        # Fetch collaborators from GitHub API
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/collaborators",
                headers=headers
            )
            
            if response.status_code != 200:
                raise ValueError(f"GitHub API error: {response.status_code} - {response.text}")
            
            github_collaborators = response.json()
        
        logger.info(f"📥 Fetched {len(github_collaborators)} collaborators from GitHub")
        
        synced_count = 0
        errors = []
        
        for github_user in github_collaborators:
            try:
                # 1. Upsert collaborator
                collaborator_id = await upsert_collaborator(github_user)
                
                # 2. Link to repository
                await link_collaborator_to_repository(
                    repository_id, 
                    collaborator_id, 
                    {**github_user, "repository_owner": owner}
                )
                
                synced_count += 1
                
            except Exception as e:
                error_msg = f"Failed to sync collaborator {github_user.get('login', 'unknown')}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        result = {
            "status": "success" if synced_count > 0 else "partial",
            "repository": f"{owner}/{repo}",
            "total_fetched": len(github_collaborators),
            "synced_count": synced_count,
            "errors": errors
        }
        
        logger.info(f"✅ Sync completed for {owner}/{repo}: {synced_count}/{len(github_collaborators)} successful")
        return result
        
    except Exception as e:
        logger.error(f"💥 Error syncing collaborators for {owner}/{repo}: {e}")
        return {
            "status": "error",
            "repository": f"{owner}/{repo}",
            "error": str(e)
        }

async def get_collaborators_with_fallback(owner: str, repo: str, github_token: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get collaborators for a repository with fallback to GitHub API if none synced
    
    Args:
        owner: Repository owner
        repo: Repository name
        github_token: Optional GitHub token for fallback
        
    Returns:
        List of collaborator data
    """
    try:
        logger.info(f"🔍 Getting collaborators for {owner}/{repo}")
        
        # Get repository ID
        repo_query = select(repositories.c.id).where(
            (repositories.c.owner == owner) &
            (repositories.c.name == repo)
        )
        repo_result = await database.fetch_one(repo_query)
        if not repo_result:
            logger.warning(f"Repository {owner}/{repo} not found in database - returning empty list")
            return []
        
        repo_id = repo_result.id
        
        # Try to get synced collaborators first
        collaborators_list = await get_collaborators_by_repo(repo_id)
        
        if collaborators_list:
            logger.info(f"✅ Found {len(collaborators_list)} synced collaborators")
            return collaborators_list
        
        # Fallback: try to fetch from GitHub directly if token provided
        if github_token:
            logger.info(f"🔄 No synced collaborators found, trying GitHub API fallback")
            
            try:
                headers = {
                    "Authorization": f"Bearer {github_token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28"
                }
                
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        f"https://api.github.com/repos/{owner}/{repo}/collaborators",
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        github_collaborators = response.json()
                        
                        # Convert to our format for frontend compatibility
                        fallback_collaborators = []
                        for i, github_user in enumerate(github_collaborators):
                            fallback_collab = {
                                "id": f"fallback_{i}",
                                "repository_id": repo_id,
                                "github_user_id": github_user.get("id"),
                                "github_username": github_user.get("login"),
                                "login": github_user.get("login"),
                                "role": "unknown",
                                "is_owner": False,  # We can't determine this easily
                                "type": "Collaborator (Live)",
                                "display_name": github_user.get("name"),
                                "avatar_url": github_user.get("avatar_url"),
                                "github_profile_url": github_user.get("html_url"),
                                "sync_status": "live_fallback"
                            }
                            fallback_collaborators.append(fallback_collab)
                        
                        logger.info(f"📡 Retrieved {len(fallback_collaborators)} collaborators via GitHub API fallback")
                        return fallback_collaborators
                        
            except Exception as e:
                logger.error(f"GitHub API fallback failed: {e}")
        
        logger.warning(f"No collaborators found for {owner}/{repo} (synced or fallback)")
        return []
        
    except Exception as e:
        logger.error(f"Error getting collaborators with fallback for {owner}/{repo}: {e}")
        return []
