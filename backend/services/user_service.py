from db.models.users import users
from sqlalchemy import select
from db.database import database

async def get_user_id_by_github_username(username: str):
    query = select(users).where(users.c.github_username == username)
    result = await database.fetch_one(query)
    if result:
        return result.id
    return None
