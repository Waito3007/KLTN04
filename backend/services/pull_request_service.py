from db.models.pull_requests import pull_requests
from sqlalchemy import select, insert, update, func
from db.database import database
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def parse_github_datetime(date_str):
    """Convert GitHub API datetime string to Python datetime object"""
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

async def get_pull_request_by_github_id(github_id: int):
    """Lấy pull request từ github_id"""
    query = select(pull_requests).where(pull_requests.c.github_id == github_id)
    result = await database.fetch_one(query)
    return result

async def save_pull_request(pr_data):
    """
    Lưu hoặc cập nhật thông tin pull request
    
    Args:
        pr_data (dict): Thông tin pull request từ GitHub API
    
    Returns:
        int: pull_request_id
    """
    try:
        # Kiểm tra xem pull request đã tồn tại chưa
        existing_pr = await get_pull_request_by_github_id(pr_data["github_id"])

        if existing_pr:
            # Nếu đã tồn tại, cập nhật thông tin
            query = (
                update(pull_requests)
                .where(pull_requests.c.github_id == pr_data["github_id"])
                .values(
                    title=pr_data.get("title"),
                    description=pr_data.get("description"),
                    state=pr_data.get("state"),
                    updated_at=parse_github_datetime(pr_data.get("updated_at"))
                )
            )
            await database.execute(query)
            logger.info(f"Updated pull request: {pr_data.get('title')}")
            return existing_pr.id
        else:
            # Nếu chưa tồn tại, thêm mới
            query = insert(pull_requests).values(
                github_id=pr_data["github_id"],
                title=pr_data.get("title"),
                description=pr_data.get("description"),
                state=pr_data.get("state"),
                repo_id=pr_data["repo_id"],
                created_at=parse_github_datetime(pr_data.get("created_at")),
                updated_at=parse_github_datetime(pr_data.get("updated_at"))
            )
            
            result = await database.execute(query)
            logger.info(f"Created new pull request: {pr_data.get('title')}")
            return result
            
    except Exception as e:
        logger.error(f"Error saving pull request {pr_data.get('title')}: {e}")
        raise e

async def get_pull_requests_by_repo_id(repo_id: int):
    """Lấy danh sách pull requests của repository"""
    query = select(pull_requests).where(pull_requests.c.repo_id == repo_id)
    results = await database.fetch_all(query)
    return results
