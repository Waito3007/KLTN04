from db.models.users import users
from sqlalchemy import select, insert, update, func
from db.database import database

async def get_user_id_by_github_username(username: str):
    query = select(users).where(users.c.github_username == username)
    result = await database.fetch_one(query)
    if result:
        return result.id
    return None

async def save_user(user_data):
    # Kiểm tra xem người dùng đã tồn tại chưa
    query = select(users).where(users.c.github_id == user_data["github_id"])
    existing_user = await database.fetch_one(query)

    if existing_user:
        # Nếu đã tồn tại, cập nhật thông tin
        query = (
            update(users)
            .where(users.c.github_id == user_data["github_id"])
            .values(
                github_username=user_data["github_username"],
                email=user_data["email"],
                avatar_url=user_data["avatar_url"],
                updated_at=func.now()  # Cập nhật thời gian
            )
        )
    else:
        # Nếu chưa tồn tại, thêm mới
        query = insert(users).values(
            github_id=user_data["github_id"],
            github_username=user_data["github_username"],
            email=user_data["email"],
            avatar_url=user_data["avatar_url"]
        )

    # Thực thi truy vấn
    await database.execute(query)
