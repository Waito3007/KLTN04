from db.models.commits import commits
from db.database import database
from sqlalchemy import select
from datetime import datetime

async def save_commit(commit_data):
    # Check xem commit đã tồn tại chưa
    query = select(commits).where(commits.c.sha == commit_data["sha"])
    result = await database.fetch_one(query)
    if result:
        return  # Đã tồn tại → bỏ qua

    insert_query = commits.insert().values(
        sha=commit_data["sha"],
        message=commit_data["message"],
        author_id=commit_data["author_id"],
        repo_id=commit_data["repo_id"],
        committed_at=commit_data["committed_at"],
        insertions=commit_data.get("insertions", 0),
        deletions=commit_data.get("deletions", 0),
        files_changed=commit_data.get("files_changed", 0),
        ai_label=None,
        created_at=datetime.utcnow()
    )
    await database.execute(insert_query)
