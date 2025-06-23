from db.models.users import users
from sqlalchemy import select, insert, update, func
from db.database import database
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

def parse_github_datetime(date_str: Optional[str]) -> Optional[datetime]:
    """
    Convert GitHub API datetime string to Python datetime object
    
    Args:
        date_str: GitHub datetime string in ISO format (e.g., '2021-03-06T14:28:54Z')
    
    Returns:
        datetime object or None if parsing fails
    """
    if not date_str:
        return None
    
    try:
        # GitHub datetime format: 2021-03-06T14:28:54Z
        if date_str.endswith('Z'):
            date_str = date_str[:-1] + '+00:00'  # Replace Z with +00:00 for proper parsing
        
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to parse datetime '{date_str}': {e}")
        return None

async def get_user_id_by_github_username(username: str):
    """Lấy user_id từ github_username"""
    query = select(users).where(users.c.github_username == username)
    result = await database.fetch_one(query)
    if result:
        return result.id
    return None

async def get_user_by_github_id(github_id: int):
    """Lấy user từ github_id"""
    query = select(users).where(users.c.github_id == github_id)
    result = await database.fetch_one(query)
    return result

async def save_user(user_data):
    """
    Lưu hoặc cập nhật thông tin user
    
    Args:
        user_data (dict): Thông tin user từ GitHub API
    
    Returns:
        int: user_id
    """
    try:
        # Convert datetime strings to datetime objects
        github_created_at = parse_github_datetime(user_data.get("github_created_at"))
        
        # Kiểm tra xem người dùng đã tồn tại chưa
        existing_user = await get_user_by_github_id(user_data["github_id"])

        if existing_user:
            # Nếu đã tồn tại, cập nhật thông tin
            query = (
                update(users)
                .where(users.c.github_id == user_data["github_id"])
                .values(
                    github_username=user_data.get("github_username"),
                    email=user_data.get("email"),
                    display_name=user_data.get("display_name"),
                    full_name=user_data.get("full_name"),
                    avatar_url=user_data.get("avatar_url"),
                    bio=user_data.get("bio"),
                    location=user_data.get("location"),
                    company=user_data.get("company"),
                    blog=user_data.get("blog"),
                    twitter_username=user_data.get("twitter_username"),
                    github_profile_url=user_data.get("github_profile_url"),
                    repos_url=user_data.get("repos_url"),
                    github_created_at=github_created_at,
                    last_synced=func.now(),
                    updated_at=func.now()
                )
            )
            await database.execute(query)
            logger.info(f"Updated user: {user_data.get('github_username')}")
            return existing_user.id
        else:
            # Nếu chưa tồn tại, thêm mới
            query = insert(users).values(
                github_id=user_data["github_id"],
                github_username=user_data.get("github_username"),
                email=user_data.get("email"),
                display_name=user_data.get("display_name"),
                full_name=user_data.get("full_name"),
                avatar_url=user_data.get("avatar_url"),
                bio=user_data.get("bio"),
                location=user_data.get("location"),
                company=user_data.get("company"),
                blog=user_data.get("blog"),
                twitter_username=user_data.get("twitter_username"),
                github_profile_url=user_data.get("github_profile_url"),
                repos_url=user_data.get("repos_url"),
                is_active=True,
                is_verified=False,
                github_created_at=github_created_at,
                last_synced=func.now(),
                created_at=func.now(),
                updated_at=func.now()
            )
            
            result = await database.execute(query)
            user_id = result
            logger.info(f"Created new user: {user_data.get('github_username')}")
            return user_id
            
    except Exception as e:
        logger.error(f"Error saving user {user_data.get('github_username')}: {e}")
        raise e
