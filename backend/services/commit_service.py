from db.models.commits import commits
from db.database import database
from sqlalchemy import select

async def save_repo_commits(commit_data):
    # Kiểm tra commit đã tồn tại chưa
    query = select(commits).where(commits.c.sha == commit_data["sha"])
    result = await database.fetch_one(query)
    if result:
        return  # Commit đã tồn tại, không lưu lại

    # Lưu commit mới
    query = commits.insert().values(
        sha=commit_data["sha"],
        message=commit_data["message"],
        author_name=commit_data["author_name"],
        author_email=commit_data["author_email"],
        date=commit_data["date"],
        repo_id=commit_data["repo_id"],
    )
    await database.execute(query)