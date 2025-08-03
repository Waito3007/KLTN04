from db.database import database
from db.models.issues import issues
from sqlalchemy import select, insert
from datetime import datetime
import logging
from utils.datetime_utils import normalize_github_datetime

logger = logging.getLogger(__name__)

# Backward compatibility - deprecated function
def parse_github_datetime(date_str):
    """Convert GitHub API datetime string to Python datetime object"""
    logger.warning("parse_github_datetime is deprecated, use normalize_github_datetime instead")
    return normalize_github_datetime(date_str)

async def get_issue_by_github_id(github_id: int):
    """Get issue by GitHub ID"""
    query = select(issues).where(issues.c.github_id == github_id)
    result = await database.fetch_one(query)
    return result

# Lưu một issue duy nhất
async def save_issue(issue_data):
    """Save issue with proper datetime conversion"""
    try:
        # Kiểm tra issue đã tồn tại chưa
        existing_issue = await get_issue_by_github_id(issue_data["github_id"])
        
        if existing_issue:
            logger.info(f"Issue {issue_data['github_id']} already exists, skipping")
            return existing_issue.id

        # Convert datetime strings
        issue_entry = {
            **issue_data,
            "created_at": normalize_github_datetime(issue_data.get("created_at")),
            "updated_at": normalize_github_datetime(issue_data.get("updated_at"))
        }

        query = insert(issues).values(issue_entry)
        result = await database.execute(query)
        logger.info(f"Created new issue: {issue_data['title']}")
        return result
        
    except Exception as e:
        logger.error(f"Error saving issue {issue_data.get('title')}: {e}")
        raise e

# Lưu danh sách nhiều issue
async def save_issues(issue_list):
    """Save multiple issues"""
    for issue in issue_list:
        await save_issue(issue)
