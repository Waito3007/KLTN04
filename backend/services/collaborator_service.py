# backend/services/collaborator_service.py
from db.models.repository_collaborators import repository_collaborators
from db.models.users import users
from db.models.repositories import repositories
from sqlalchemy import select, join, func
from db.database import database
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

async def get_collaborators_by_repo(repo_id: int) -> List[Dict[str, Any]]:
    """
    Get all collaborators for a specific repository
    
    Args:
        repo_id: Repository ID
        
    Returns:
        List of collaborator data with user information
    """
    try:        # Join repository_collaborators with users table to get full user info
        query = (
            select(
                repository_collaborators.c.id,
                repository_collaborators.c.repository_id,
                repository_collaborators.c.user_id,
                repository_collaborators.c.role,
                repository_collaborators.c.permissions,
                repository_collaborators.c.is_owner,
                repository_collaborators.c.commits_count,
                repository_collaborators.c.issues_count,
                repository_collaborators.c.prs_count,
                repository_collaborators.c.joined_at,
                repository_collaborators.c.last_synced.label('collab_last_synced'),
                # User info
                users.c.github_id,
                users.c.github_username,
                users.c.display_name,
                users.c.full_name,
                users.c.email,
                users.c.avatar_url,
                users.c.bio,
                users.c.location,
                users.c.company,
                users.c.blog,
                users.c.twitter_username,
                users.c.github_profile_url,
                users.c.github_created_at
            )
            .select_from(
                repository_collaborators.join(
                    users,
                    repository_collaborators.c.user_id == users.c.id,
                    isouter=True
                )
            )
            .where(repository_collaborators.c.repository_id == repo_id)
            .order_by(repository_collaborators.c.is_owner.desc(), repository_collaborators.c.role.desc())        )
        
        results = await database.fetch_all(query)
        
        collaborators = []
        for row in results:
            collab_data = {
                "id": row.id,
                "repository_id": row.repository_id,
                "user_id": row.user_id,
                "github_id": row.github_id,
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
                # User info
                "display_name": row.display_name,
                "full_name": row.full_name,
                "email": row.email,
                "avatar_url": row.avatar_url,
                "bio": row.bio,
                "location": row.location,
                "company": row.company,
                "blog": row.blog,
                "twitter_username": row.twitter_username,
                "github_profile_url": row.github_profile_url,
                "github_created_at": row.github_created_at
            }
            collaborators.append(collab_data)
        
        logger.info(f"Retrieved {len(collaborators)} collaborators for repo_id {repo_id}")
        return collaborators
        
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
        from sqlalchemy import update
        
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
    
    Args:
        owner: Repository owner (GitHub username)
        repo: Repository name
        
    Returns:
        List of collaborator data with user information
    """
    try:
        logger.info(f"🔍 Looking for collaborators for repository: {owner}/{repo}")
        
        # First get the repository ID
        repo_query = select(repositories.c.id).where(
            (repositories.c.owner == owner) &
            (repositories.c.name == repo)
        )
        repo_result = await database.fetch_one(repo_query)
        
        if not repo_result:
            logger.warning(f"❌ Repository {owner}/{repo} not found in database")
            # Let's also check what repositories exist
            all_repos_query = select(repositories.c.owner, repositories.c.name).limit(10)
            all_repos = await database.fetch_all(all_repos_query)
            logger.info(f"📋 Available repositories in database:")
            for r in all_repos:
                logger.info(f"   - {r.owner}/{r.name}")
            return []
        
        repo_id = repo_result.id
        logger.info(f"✅ Found repository {owner}/{repo} with ID: {repo_id}")
        
        # Get collaborators with user info using the existing function
        collaborators = await get_collaborators_by_repo(repo_id)
        logger.info(f"👥 Found {len(collaborators)} collaborators for {owner}/{repo}")
        return collaborators
        
    except Exception as e:
        logger.error(f"💥 Error getting collaborators with user info for {owner}/{repo}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []
