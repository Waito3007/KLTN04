from db.models.repository_collaborators import repository_collaborators
from sqlalchemy import select, insert, update, func
from db.database import database
import logging

logger = logging.getLogger(__name__)

async def get_repository_collaborator(repository_id: int, user_id: int):
    """Lấy thông tin collaborator của repository"""
    query = select(repository_collaborators).where(
        repository_collaborators.c.repository_id == repository_id,
        repository_collaborators.c.user_id == user_id
    )
    result = await database.fetch_one(query)
    return result

async def save_repository_collaborator(collaborator_data):
    """
    Lưu hoặc cập nhật thông tin repository collaborator
    
    Args:
        collaborator_data (dict): Thông tin collaborator
            - repository_id: ID của repository
            - user_id: ID của user
            - role: vai trò (admin, write, read, etc.)
            - permissions: quyền hạn
            - is_owner: có phải owner không
    
    Returns:
        int: collaborator_id
    """
    try:
        # Kiểm tra xem collaborator đã tồn tại chưa
        existing_collaborator = await get_repository_collaborator(
            collaborator_data["repository_id"], 
            collaborator_data["user_id"]
        )

        if existing_collaborator:
            # Nếu đã tồn tại, cập nhật thông tin
            query = (
                update(repository_collaborators)
                .where(
                    repository_collaborators.c.repository_id == collaborator_data["repository_id"],
                    repository_collaborators.c.user_id == collaborator_data["user_id"]
                )
                .values(
                    role=collaborator_data.get("role", "read"),
                    permissions=collaborator_data.get("permissions"),
                    is_owner=collaborator_data.get("is_owner", False),
                    invitation_status=collaborator_data.get("invitation_status", "accepted"),
                    last_synced=func.now()
                )
            )
            await database.execute(query)
            logger.info(f"Updated repository collaborator: repo_id={collaborator_data['repository_id']}, user_id={collaborator_data['user_id']}")
            return existing_collaborator.id
        else:
            # Nếu chưa tồn tại, thêm mới
            query = insert(repository_collaborators).values(
                repository_id=collaborator_data["repository_id"],
                user_id=collaborator_data["user_id"],
                role=collaborator_data.get("role", "read"),
                permissions=collaborator_data.get("permissions"),
                is_owner=collaborator_data.get("is_owner", False),
                joined_at=collaborator_data.get("joined_at"),
                invited_by=collaborator_data.get("invited_by"),
                invitation_status=collaborator_data.get("invitation_status", "accepted"),
                commits_count=collaborator_data.get("commits_count", 0),
                issues_count=collaborator_data.get("issues_count", 0),
                prs_count=collaborator_data.get("prs_count", 0),
                last_activity=collaborator_data.get("last_activity"),
                last_synced=func.now()
            )
            
            result = await database.execute(query)
            logger.info(f"Created new repository collaborator: repo_id={collaborator_data['repository_id']}, user_id={collaborator_data['user_id']}")
            return result
            
    except Exception as e:
        logger.error(f"Error saving repository collaborator: {e}")
        raise e

async def get_collaborators_by_repository_id(repository_id: int):
    """Lấy danh sách collaborators của repository"""
    query = select(repository_collaborators).where(
        repository_collaborators.c.repository_id == repository_id
    )
    results = await database.fetch_all(query)
    return results
