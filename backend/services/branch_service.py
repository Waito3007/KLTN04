from db.models.branches import branches
from db.database import database
from services.repo_service import get_repo_id_by_owner_and_name
from sqlalchemy import select, delete
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

async def save_branch(branch_data):
    """Save single branch to database with full data"""
    query = branches.insert().values(
        name=branch_data["name"],
        repo_id=branch_data["repo_id"],
        # Thêm các fields mới
        sha=branch_data.get("sha"),
        is_default=branch_data.get("is_default", False),
        is_protected=branch_data.get("is_protected", False),
        creator_name=branch_data.get("creator_name"),
        last_committer_name=branch_data.get("last_committer_name"),
        last_commit_date=branch_data.get("last_commit_date"),
        commits_count=branch_data.get("commits_count"),
        contributors_count=branch_data.get("contributors_count"),
        created_at=branch_data.get("created_at"),
    )
    await database.execute(query)

async def save_multiple_branches(repo_id: int, branches_list: list, default_branch: str = "main"):
    """Save multiple branches efficiently with full data"""
    if not branches_list:
        return 0
    
    # Prepare batch data với dữ liệu đã được chuẩn hóa từ API layer
    batch_data = []
    for branch in branches_list:
        branch_data = {
            "name": branch["name"],
            "repo_id": repo_id,
            # Dữ liệu đã được chuẩn hóa từ sync.py
            "sha": branch.get("sha"),
            "is_default": branch.get("is_default", branch["name"] == default_branch),
            "is_protected": branch.get("is_protected", False),
            
            # Các trường sẽ được thêm sau khi có thêm API calls
            "last_commit_date": branch.get("last_commit_date"),
            "creator_name": branch.get("creator_name"),
            "last_committer_name": branch.get("last_committer_name"),
            "commits_count": branch.get("commits_count"),
            "contributors_count": branch.get("contributors_count"),
            "created_at": branch.get("created_at"),
        }
        batch_data.append(branch_data)
    
    # Batch insert
    if batch_data:
        query = branches.insert()
        await database.execute_many(query, batch_data)
    
    return len(batch_data)

async def get_branches_by_repo_id(repo_id: int):
    """Get all branches for a repository"""
    query = select(branches).where(branches.c.repo_id == repo_id)
    result = await database.fetch_all(query)
    return [dict(row) for row in result]

async def get_branches_by_owner_repo(owner: str, repo_name: str):
    """Get branches by owner/repo name"""
    repo_id = await get_repo_id_by_owner_and_name(owner, repo_name)
    if not repo_id:
        return []
    
    return await get_branches_by_repo_id(repo_id)

async def delete_branches_by_repo_id(repo_id: int):
    """Delete branches for a repository, handling foreign key constraints safely"""
    from db.models.commits import commits
    
    # First, update commits to remove branch_id references
    update_commits_query = commits.update().where(
        commits.c.repo_id == repo_id
    ).values(branch_id=None)
    
    await database.execute(update_commits_query)
    
    # Then delete branches
    query = delete(branches).where(branches.c.repo_id == repo_id)
    result = await database.execute(query)
    
    logger.info(f"Updated commits and deleted {result} branches for repo_id {repo_id}")
    return result

async def sync_branches_for_repo(repo_id: int, branches_data: list, default_branch: str = "main", replace_existing: bool = False):
    """
    Sync branches for a repository with full branch data
    Args:
        repo_id: Repository ID
        branches_data: List of branch data from GitHub API
        default_branch: Name of default branch (from repo data)
        replace_existing: If True, delete existing branches first
    """
    if replace_existing:
        await delete_branches_by_repo_id(repo_id)
    
    saved_count = await save_multiple_branches(repo_id, branches_data, default_branch)
    logger.info(f"Synced {saved_count} branches for repo_id {repo_id}")
    
    return saved_count

async def get_branch_statistics(repo_id: int):
    """
    Lấy thống kê về các branches của repository
    
    Args:
        repo_id: ID của repository
    
    Returns:
        dict: Thống kê branches
    """
    from sqlalchemy import func, case
    
    query = select([
        func.count().label('total_branches'),
        func.count(case([(branches.c.is_default == True, 1)])).label('default_branches'),
        func.count(case([(branches.c.is_protected == True, 1)])).label('protected_branches'),
        func.count(case([(branches.c.last_commit_date.isnot(None), 1)])).label('branches_with_commits'),
        func.avg(branches.c.commits_count).label('avg_commits_per_branch'),
        func.max(branches.c.last_commit_date).label('last_activity')
    ]).where(branches.c.repo_id == repo_id)
    
    result = await database.fetch_one(query)
    
    if result:
        return {
            'total_branches': result['total_branches'] or 0,
            'default_branches': result['default_branches'] or 0,
            'protected_branches': result['protected_branches'] or 0,
            'branches_with_commits': result['branches_with_commits'] or 0,
            'avg_commits_per_branch': float(result['avg_commits_per_branch'] or 0),
            'last_activity': result['last_activity']
        }
    
    return {
        'total_branches': 0,
        'default_branches': 0,
        'protected_branches': 0,
        'branches_with_commits': 0,
        'avg_commits_per_branch': 0,
        'last_activity': None
    }

async def find_stale_branches(repo_id: int, days_threshold: int = 90):
    """
    Tìm các branches cũ (không có commit trong X ngày)
    
    Args:
        repo_id: ID của repository
        days_threshold: Số ngày để coi là cũ
    
    Returns:
        list: Danh sách branches cũ
    """
    from datetime import datetime, timedelta
    
    cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
    
    query = select(branches).where(
        (branches.c.repo_id == repo_id) &
        (branches.c.last_commit_date < cutoff_date) &
        (branches.c.is_default == False)  # Không bao gồm default branch
    ).order_by(branches.c.last_commit_date.asc())
    
    result = await database.fetch_all(query)
    return [dict(row) for row in result]

async def get_most_active_branches(repo_id: int, limit: int = 10):
    """
    Lấy các branches hoạt động nhiều nhất (theo số commits)
    
    Args:
        repo_id: ID của repository
        limit: Số lượng branches trả về
    
    Returns:
        list: Danh sách branches hoạt động nhất
    """
    query = select(branches).where(
        (branches.c.repo_id == repo_id) &
        (branches.c.commits_count.isnot(None))
    ).order_by(branches.c.commits_count.desc()).limit(limit)
    
    result = await database.fetch_all(query)
    return [dict(row) for row in result]

async def update_branch_metadata(repo_id: int, branch_name: str, metadata: dict):
    """
    Cập nhật metadata của một branch cụ thể
    
    Args:
        repo_id: ID của repository
        branch_name: Tên branch
        metadata: Dictionary chứa dữ liệu cần cập nhật
    
    Returns:
        bool: True nếu cập nhật thành công
    """
    from sqlalchemy import update
    
    # Chỉ cho phép cập nhật một số fields nhất định
    allowed_fields = {
        'sha', 'is_protected', 'last_commit_date', 'last_committer_name',
        'commits_count', 'contributors_count', 'creator_name'
    }
    
    update_data = {k: v for k, v in metadata.items() if k in allowed_fields}
    
    if not update_data:
        return False
    
    query = update(branches).where(
        (branches.c.repo_id == repo_id) & 
        (branches.c.name == branch_name)
    ).values(**update_data)
    
    result = await database.execute(query)
    return result is not None and result > 0