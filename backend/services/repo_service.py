# backend/services/repo_service.py
from .github_service import fetch_from_github

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
