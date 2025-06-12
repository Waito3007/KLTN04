# backend/services/repo_service.py
from .github_service import fetch_from_github
from db.models.repositories import repositories
from sqlalchemy import select, update
from sqlalchemy.sql import func
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


async def save_repository(repo_entry):
    # Kiểm tra xem repository đã tồn tại chưa
    query = select(repositories).where(repositories.c.github_id == repo_entry["github_id"])
    existing_repo = await database.fetch_one(query)

    if existing_repo:
        # Nếu repository đã tồn tại, cập nhật thông tin (nếu cần)
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
                updated_at=func.now(),
            )
        )
        await database.execute(update_query)
    else:
        # Nếu repository chưa tồn tại, chèn mới
        query = repositories.insert().values(repo_entry)
        await database.execute(query)


async def get_repo_by_owner_and_name(owner: str, repo: str):
    """Get repository information by owner and name"""
    query = select(repositories).where(
        repositories.c.owner == owner,
        repositories.c.name == repo
    )
    result = await database.fetch_one(query)
    if result:
        return dict(result)
    return None


async def get_github_repo_id(owner: str, repo: str):
    """Get GitHub repository ID from GitHub API"""
    try:
        url = f"/repos/{owner}/{repo}"
        data = await fetch_from_github(url)
        return data.get("id")  # GitHub repo ID
    except Exception as e:
        print(f"Error getting GitHub repo ID: {e}")
        return None