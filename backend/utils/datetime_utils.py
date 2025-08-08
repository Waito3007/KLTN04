# backend/utils/datetime_utils.py
from datetime import datetime, timezone
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def normalize_github_datetime(dt_string: Optional[str]) -> Optional[datetime]:
    """
    Chuẩn hóa datetime từ GitHub API về format phù hợp với database
    
    Args:
        dt_string: ISO datetime string từ GitHub API (có thể có 'Z' suffix)
        
    Returns:
        datetime object đã normalize (UTC, no timezone info) hoặc None
    """
    if not dt_string:
        return None
    
    try:
        # GitHub API trả về format: "2024-06-30T09:57:23Z"
        # Chuyển 'Z' thành '+00:00' để parse được
        if dt_string.endswith('Z'):
            dt_string = dt_string.replace('Z', '+00:00')
        
        # Parse datetime với timezone
        dt = datetime.fromisoformat(dt_string)
        
        # Convert về UTC và remove timezone info để consistent với DB
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        
        return dt
        
    except Exception as e:
        logger.error(f"Error parsing datetime '{dt_string}': {e}")
        return None

def normalize_datetime(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Normalize datetime object to timezone-naive UTC for consistent database storage
    
    Args:
        dt: datetime object (có thể có hoặc không có timezone)
        
    Returns:
        datetime object đã normalize (UTC, no timezone info) hoặc None
    """
    if dt is None:
        return None
    
    try:
        if dt.tzinfo is not None:
            # Convert timezone-aware to UTC and make timezone-naive
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            # Already timezone-naive, assume it's UTC
            return dt
    except Exception as e:
        logger.error(f"Error normalizing datetime '{dt}': {e}")
        return None
