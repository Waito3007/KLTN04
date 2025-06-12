from db.models.commits import commits
from db.database import database
from sqlalchemy import select

async def save_commit(commit_data):
    # Kiểm tra commit đã tồn tại chưa
    query = select(commits).where(commits.c.sha == commit_data["sha"])
    existing_commit = await database.fetch_one(query)

    if existing_commit:
        return  # Bỏ qua nếu commit đã tồn tại

    # Chèn commit mới
    query = commits.insert().values(commit_data)
    await database.execute(query)

async def get_commits_by_repo(owner: str, repo: str, limit: int = 100):
    """Get commits by repository owner and name"""
    query = select(commits).where(
        commits.c.repo_owner == owner,
        commits.c.repo_name == repo
    ).limit(limit)
    return await database.fetch_all(query)

async def get_repo_by_owner_and_name(owner: str, repo: str):
    """Get repository information by owner and name"""
    # Dummy implementation - replace with actual repo service
    return {"owner": owner, "name": repo}