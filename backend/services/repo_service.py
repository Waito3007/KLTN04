# backend/services/repo_service.py
from .github_service import fetch_from_github
from db.models.repositories import repositories
from sqlalchemy import select, update
from sqlalchemy.sql import func
from db.database import database

async def get_repository(owner: str, repo_name: str, id_only: bool = False):
    """Get repository from DB - flexible return"""
    query = select(repositories).where(
        repositories.c.owner == owner,
        repositories.c.name == repo_name
    )
    result = await database.fetch_one(query)
    if result:
        return result.id if id_only else dict(result)
    return None

async def get_repo_id_by_owner_and_name(owner: str, repo_name: str):
    """Backward compatibility wrapper"""
    return await get_repository(owner, repo_name, id_only=True)

async def get_repo_by_owner_and_name(owner: str, repo_name: str):
    """Backward compatibility wrapper"""
    return await get_repository(owner, repo_name, id_only=False)

async def save_repository(repo_entry):
    """Save or update repository"""    # Check if exists
    query = select(repositories).where(repositories.c.github_id == repo_entry["github_id"])
    existing_repo = await database.fetch_one(query)
    
    if existing_repo:
        # Update existing
        update_query = (
            update(repositories)
            .where(repositories.c.github_id == repo_entry["github_id"])
            .values(
                name=repo_entry["name"],
                owner=repo_entry["owner"],
                description=repo_entry["description"],
                stars=repo_entry["stars"],
                forks=repo_entry["forks"],
                language=repo_entry["language"],
                open_issues=repo_entry["open_issues"],
                url=repo_entry["url"],
                # Bổ sung các fields mới
                full_name=repo_entry.get("full_name"),
                clone_url=repo_entry.get("clone_url"),
                is_private=repo_entry.get("is_private", False),
                is_fork=repo_entry.get("is_fork", False),
                default_branch=repo_entry.get("default_branch", "main"),
                sync_status=repo_entry.get("sync_status", "completed"),
                updated_at=func.now(),
            )
        )
        await database.execute(update_query)
    else:
        # Insert new
        query = repositories.insert().values(repo_entry)
        await database.execute(query)

async def fetch_repo_from_github(owner: str, repo: str):
    """Fetch repository data from GitHub API"""
    url = f"/repos/{owner}/{repo}"
    data = await fetch_from_github(url)
    
    return {
        "github_id": data.get("id"),
        "name": data.get("name"),
        "owner": data.get("owner", {}).get("login"),
        "description": data.get("description"),
        "stars": data.get("stargazers_count"),
        "forks": data.get("forks_count"),
        "language": data.get("language"),
        "open_issues": data.get("open_issues_count"),
        "url": data.get("html_url"),
        # Bổ sung các fields mới
        "full_name": data.get("full_name"),
        "clone_url": data.get("clone_url"),
        "is_private": data.get("private", False),
        "is_fork": data.get("fork", False),
        "default_branch": data.get("default_branch", "main"),
        "sync_status": "completed",
    }

async def fetch_repo_from_database(owner: str, repo_name: str):
    """Fetch repository data from database (similar to GitHub API format)"""
    repo_data = await get_repository(owner, repo_name, id_only=False)
    
    if not repo_data:
        return None
    
    # Format similar to GitHub API response
    return {
        "id": repo_data.get("github_id"),
        "name": repo_data.get("name"),
        "full_name": repo_data.get("full_name") or f"{repo_data.get('owner')}/{repo_data.get('name')}",
        "owner": {
            "login": repo_data.get("owner")
        },
        "description": repo_data.get("description"),
        "stargazers_count": repo_data.get("stars", 0),
        "forks_count": repo_data.get("forks", 0),
        "language": repo_data.get("language"),
        "open_issues_count": repo_data.get("open_issues", 0),
        "html_url": repo_data.get("url"),
        "clone_url": repo_data.get("clone_url"),
        "private": repo_data.get("is_private", False),
        "fork": repo_data.get("is_fork", False),
        "default_branch": repo_data.get("default_branch", "main"),
        "sync_status": repo_data.get("sync_status"),
        "last_synced": repo_data.get("last_synced"),
        "created_at": repo_data.get("created_at"),
        "updated_at": repo_data.get("updated_at"),
    }

async def get_user_repos_from_database(user_id: int = None, limit: int = 100, offset: int = 0):
    """Get user repositories from database (similar to GitHub API but from DB)"""
    query = select(repositories)
    
    # Filter by user if provided
    if user_id:
        query = query.where(repositories.c.user_id == user_id)
    
    # Add pagination
    query = query.limit(limit).offset(offset).order_by(repositories.c.updated_at.desc())
    
    results = await database.fetch_all(query)
    
    # Format results similar to GitHub API response
    repos = []
    for repo in results:
        repos.append({
            "id": repo.github_id,
            "name": repo.name,
            "full_name": repo.full_name or f"{repo.owner}/{repo.name}",
            "owner": {
                "login": repo.owner
            },
            "description": repo.description,
            "stargazers_count": repo.stars or 0,
            "forks_count": repo.forks or 0,
            "language": repo.language,
            "open_issues_count": repo.open_issues or 0,
            "html_url": repo.url,
            "clone_url": repo.clone_url,
            "private": repo.is_private or False,
            "fork": repo.is_fork or False,
            "default_branch": repo.default_branch or "main",
            "sync_status": repo.sync_status,
            "last_synced": repo.last_synced,
            "created_at": repo.created_at,
            "updated_at": repo.updated_at
        })
    
    return repos

async def get_repositories_by_owner(owner: str, limit: int = 100, offset: int = 0):
    """Get all repositories by owner from database"""
    query = select(repositories).where(
        repositories.c.owner == owner
    ).limit(limit).offset(offset).order_by(repositories.c.updated_at.desc())
    
    results = await database.fetch_all(query)
    
    # Format results similar to GitHub API response
    repos = []
    for repo in results:
        repos.append({
            "id": repo.github_id,
            "name": repo.name,
            "full_name": repo.full_name or f"{repo.owner}/{repo.name}",
            "owner": {
                "login": repo.owner
            },
            "description": repo.description,
            "stargazers_count": repo.stars or 0,
            "forks_count": repo.forks or 0,
            "language": repo.language,
            "open_issues_count": repo.open_issues or 0,
            "html_url": repo.url,
            "clone_url": repo.clone_url,
            "private": repo.is_private or False,
            "fork": repo.is_fork or False,
            "default_branch": repo.default_branch or "main",
            "sync_status": repo.sync_status,
            "last_synced": repo.last_synced,
            "created_at": repo.created_at,
            "updated_at": repo.updated_at
        })
    
    return repos

async def get_repository_stats():
    """Get repository statistics from database"""
    query = select([
        func.count().label('total_repos'),
        func.count(func.distinct(repositories.c.owner)).label('unique_owners'),
        func.count(func.distinct(repositories.c.language)).label('unique_languages'),
        func.sum(repositories.c.stars).label('total_stars'),
        func.sum(repositories.c.forks).label('total_forks'),
        func.avg(repositories.c.stars).label('avg_stars'),
        func.max(repositories.c.updated_at).label('last_updated')
    ])
    
    result = await database.fetch_one(query)
    
    return {
        'total_repositories': result['total_repos'] or 0,
        'unique_owners': result['unique_owners'] or 0,
        'unique_languages': result['unique_languages'] or 0,
        'total_stars': result['total_stars'] or 0,
        'total_forks': result['total_forks'] or 0,
        'average_stars': float(result['avg_stars'] or 0),
        'last_updated': result['last_updated']
    }