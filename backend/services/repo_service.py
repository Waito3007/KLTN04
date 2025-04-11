# backend/services/repo_service.py
from .github_service import fetch_from_github
from db.models.repositories import repositories
from sqlalchemy import select
from db.database import database

async def get_repo_data(owner: str, repo: str):
    url = f"/repos/{owner}/{repo}"
    data = await fetch_from_github(url)

    # Optionally: lọc data bạn muốn trả về
    return {
        "name": data.get("name"),
        "full_name": data.get("full_name"),
        "description": data.get("description"),
        "owner": data.get("owner", {}).get("login"),
        "stars": data.get("stargazers_count"),
        "forks": data.get("forks_count"),
        "watchers": data.get("watchers_count"),
        "language": data.get("language"),
        "open_issues": data.get("open_issues_count"),
        "url": data.get("html_url"),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
    }


async def get_repo_id_by_owner_and_name(owner: str, repo_name: str):
    query = select(repositories).where(
        repositories.c.owner == owner,
        repositories.c.name == repo_name
    )
    result = await database.fetch_one(query)
    if result:
        return result.id
    return None
