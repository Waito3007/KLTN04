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