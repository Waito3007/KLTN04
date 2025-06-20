from db.models.commits import commits
from db.database import database
from sqlalchemy import select, insert, update, and_
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

async def save_commit(commit_data):
    """Save commit with full data model support including new fields"""
    try:
        # Kiểm tra commit đã tồn tại chưa
        query = select(commits).where(commits.c.sha == commit_data["sha"])
        existing_commit = await database.fetch_one(query)

        if existing_commit:
            logger.info(f"Commit {commit_data['sha']} already exists, skipping")
            return existing_commit.id

        # Prepare full commit entry with all model fields
        commit_entry = {
            "sha": commit_data["sha"],
            "message": commit_data.get("message", ""),
            "author_name": commit_data.get("author_name", ""),
            "author_email": commit_data.get("author_email", ""),
            "committer_name": commit_data.get("committer_name"),
            "committer_email": commit_data.get("committer_email"),
            "repo_id": commit_data["repo_id"],
            "branch_id": commit_data.get("branch_id"),
            "branch_name": commit_data.get("branch_name"),
            "author_role_at_commit": commit_data.get("author_role_at_commit"),
            "author_permissions_at_commit": commit_data.get("author_permissions_at_commit"),
            "date": parse_github_datetime(commit_data.get("date")),
            "committer_date": parse_github_datetime(commit_data.get("committer_date")),
            "insertions": commit_data.get("insertions"),
            "deletions": commit_data.get("deletions"),
            "files_changed": commit_data.get("files_changed"),
            "parent_sha": commit_data.get("parent_sha"),
            "is_merge": commit_data.get("is_merge", False),
            "merge_from_branch": commit_data.get("merge_from_branch"),
            # author_user_id và committer_user_id sẽ được resolve sau nếu có user mapping
            "author_user_id": commit_data.get("author_user_id"),
            "committer_user_id": commit_data.get("committer_user_id"),
        }

        # Chèn commit mới
        query = insert(commits).values(commit_entry)
        result = await database.execute(query)
        logger.info(f"Created new commit: {commit_data['sha']}")
        return result
        
    except Exception as e:
        logger.error(f"Error saving commit {commit_data.get('sha')}: {e}")
        raise e

async def save_multiple_commits(commits_data: list, repo_id: int, branch_name: str = None, branch_id: int = None):
    """
    Batch save multiple commits efficiently
    
    Args:
        commits_data: List of commit data from GitHub API
        repo_id: Repository ID
        branch_name: Branch name (optional)
        branch_id: Branch ID (optional)
    
    Returns:
        int: Number of commits saved
    """
    if not commits_data:
        return 0
    
    # Prepare batch data
    batch_data = []
    existing_shas = set()
    
    # Check existing commits to avoid duplicates
    shas = [commit.get("sha") for commit in commits_data if commit.get("sha")]
    if shas:
        query = select(commits.c.sha).where(commits.c.sha.in_(shas))
        existing_results = await database.fetch_all(query)
        existing_shas = {row.sha for row in existing_results}
    
    # Process commits data
    for commit_data in commits_data:
        sha = commit_data.get("sha")
        if not sha or sha in existing_shas:
            continue
        
        # Extract commit info from GitHub API response
        commit_info = commit_data.get("commit", {})
        author_info = commit_info.get("author", {})
        committer_info = commit_info.get("committer", {})
        stats = commit_data.get("stats", {})
        
        # Check if this is a merge commit
        parents = commit_data.get("parents", [])
        is_merge = len(parents) > 1
        parent_sha = parents[0].get("sha") if parents else None
        
        commit_entry = {
            "sha": sha,
            "message": commit_info.get("message", ""),
            "author_name": author_info.get("name", ""),
            "author_email": author_info.get("email", ""),
            "committer_name": committer_info.get("name"),
            "committer_email": committer_info.get("email"),
            "repo_id": repo_id,
            "branch_id": branch_id,
            "branch_name": branch_name,
            "date": parse_github_datetime(author_info.get("date")),
            "committer_date": parse_github_datetime(committer_info.get("date")),
            "insertions": stats.get("additions"),
            "deletions": stats.get("deletions"),
            "files_changed": len(commit_data.get("files", [])) if commit_data.get("files") else None,
            "parent_sha": parent_sha,
            "is_merge": is_merge,
            # User IDs và permissions sẽ được resolve sau nếu có user service
        }
        
        batch_data.append(commit_entry)
    
    # Batch insert
    if batch_data:
        query = commits.insert()
        await database.execute_many(query, batch_data)
        logger.info(f"Batch saved {len(batch_data)} commits for repo_id {repo_id}")
    
    return len(batch_data)

async def get_commits_by_repo_id(repo_id: int, limit: int = 100, offset: int = 0):
    """Get commits by repository ID with pagination"""
    query = select(commits).where(
        commits.c.repo_id == repo_id
    ).order_by(commits.c.date.desc()).limit(limit).offset(offset)
    
    result = await database.fetch_all(query)
    return [dict(row) for row in result]

async def get_commit_by_sha(sha: str):
    """Get single commit by SHA"""
    query = select(commits).where(commits.c.sha == sha)
    result = await database.fetch_one(query)
    return dict(result) if result else None

async def get_commit_statistics(repo_id: int):
    """Get commit statistics for a repository"""
    from sqlalchemy import func
    
    try:
        # Basic stats query
        query = select(
            func.count().label('total_commits'),
            func.sum(commits.c.insertions).label('total_insertions'),
            func.sum(commits.c.deletions).label('total_deletions'),
            func.sum(commits.c.files_changed).label('total_files_changed'),
            func.count(func.distinct(commits.c.author_email)).label('unique_authors'),
            func.max(commits.c.date).label('latest_commit_date'),
            func.min(commits.c.date).label('first_commit_date')
        ).where(commits.c.repo_id == repo_id)
        
        result = await database.fetch_one(query)
        
        # Count merge commits separately
        merge_query = select(func.count()).where(
            (commits.c.repo_id == repo_id) & (commits.c.is_merge == True)
        )
        merge_count = await database.fetch_val(merge_query)
        
        if result:
            return {
                'total_commits': result['total_commits'] or 0,
                'merge_commits': merge_count or 0,
                'total_insertions': result['total_insertions'] or 0,
                'total_deletions': result['total_deletions'] or 0,
                'total_files_changed': result['total_files_changed'] or 0,
                'unique_authors': result['unique_authors'] or 0,
                'latest_commit_date': result['latest_commit_date'],
                'first_commit_date': result['first_commit_date']
            }
        
    except Exception as e:
        print(f"Error in get_commit_statistics: {e}")
        
    return {
        'total_commits': 0,
        'merge_commits': 0,
        'total_insertions': 0,
        'total_deletions': 0,
        'total_files_changed': 0,
        'unique_authors': 0,
        'latest_commit_date': None,
        'first_commit_date': None
    }

async def update_commit_user_mapping(sha: str, author_user_id: int = None, committer_user_id: int = None):
    """Update commit with user ID mappings"""
    update_data = {}
    if author_user_id:
        update_data['author_user_id'] = author_user_id
    if committer_user_id:
        update_data['committer_user_id'] = committer_user_id
    
    if update_data:
        query = update(commits).where(commits.c.sha == sha).values(**update_data)
        result = await database.execute(query)
        return result > 0
    
    return False

# ==================== NEW BRANCH-SPECIFIC COMMIT FUNCTIONS ====================

async def get_repo_id_by_owner_and_name(owner: str, name: str):
    """Get repository ID by owner and name"""
    from db.models.repositories import repositories
    
    query = select(repositories.c.id).where(
        repositories.c.owner == owner,
        repositories.c.name == name
    )
    result = await database.fetch_one(query)
    return result.id if result else None

async def get_branch_id_by_repo_and_name(repo_id: int, branch_name: str):
    """Get branch ID and validate it exists in the repository"""
    from db.models.branches import branches
    
    query = select(branches).where(
        branches.c.repo_id == repo_id,
        branches.c.name == branch_name
    )
    result = await database.fetch_one(query)
    return result if result else None

async def get_commits_by_branch_safe(repo_id: int, branch_name: str, limit: int = 100, offset: int = 0):
    """
    Lấy commits theo branch với validation đầy đủ
    """
    try:
        # 1. Verify branch exists and get branch info
        branch = await get_branch_id_by_repo_and_name(repo_id, branch_name)
        
        if not branch:
            logger.warning(f"Branch {branch_name} not found in repo_id {repo_id}")
            return []
        
        # 2. Get commits using both branch_id AND branch_name for data consistency
        query = select(commits).where(
            commits.c.repo_id == repo_id,
            commits.c.branch_id == branch.id,
            commits.c.branch_name == branch_name  # Double validation
        ).order_by(commits.c.date.desc()).limit(limit).offset(offset)
        
        commits_data = await database.fetch_all(query)
        
        logger.info(f"✅ Found {len(commits_data)} commits for branch {branch_name} in repo {repo_id}")
        return commits_data
        
    except Exception as e:
        logger.error(f"❌ Error getting commits for branch {branch_name}: {e}")
        return []

async def get_commits_by_branch_fallback(repo_id: int, branch_name: str, limit: int = 100, offset: int = 0):
    """
    Fallback method: lấy commits chỉ bằng branch_name nếu branch_id không có
    """
    try:
        query = select(commits).where(
            commits.c.repo_id == repo_id,
            commits.c.branch_name == branch_name
        ).order_by(commits.c.date.desc()).limit(limit).offset(offset)
        
        commits_data = await database.fetch_all(query)
        
        logger.info(f"⚠️ Fallback: Found {len(commits_data)} commits for branch {branch_name} using branch_name only")
        return commits_data
        
    except Exception as e:
        logger.error(f"❌ Fallback failed for branch {branch_name}: {e}")
        return []

async def get_all_branches_with_commit_stats(repo_id: int):
    """
    Lấy danh sách tất cả branches với thống kê commits
    """
    try:
        from db.models.branches import branches
        from sqlalchemy import func
        
        # Query tất cả branches với commit count
        query = select(
            branches.c.id,
            branches.c.name,
            branches.c.is_default,
            branches.c.is_protected,
            branches.c.commits_count,
            branches.c.last_commit_date,
            func.count(commits.c.id).label('actual_commit_count'),
            func.max(commits.c.date).label('latest_commit_date')
        ).select_from(
            branches.outerjoin(
                commits, 
                branches.c.id == commits.c.branch_id
            )
        ).where(
            branches.c.repo_id == repo_id
        ).group_by(
            branches.c.id,
            branches.c.name,
            branches.c.is_default,
            branches.c.is_protected,
            branches.c.commits_count,
            branches.c.last_commit_date
        ).order_by(branches.c.is_default.desc(), branches.c.name)
        
        branches_data = await database.fetch_all(query)
        
        logger.info(f"✅ Found {len(branches_data)} branches with commit stats for repo {repo_id}")
        return branches_data
        
    except Exception as e:
        logger.error(f"❌ Error getting branches with commit stats: {e}")
        return []

async def compare_commits_between_branches(repo_id: int, base_branch: str, head_branch: str, limit: int = 100):
    """
    So sánh commits giữa 2 branches
    """
    try:
        # Get commits from head branch that are not in base branch
        from sqlalchemy import and_, not_, exists
        
        # Subquery để lấy SHAs của base branch
        base_commits_subquery = select(commits.c.sha).where(
            and_(
                commits.c.repo_id == repo_id,
                commits.c.branch_name == base_branch
            )
        )
        
        # Main query: commits in head but not in base
        query = select(commits).where(
            and_(
                commits.c.repo_id == repo_id,
                commits.c.branch_name == head_branch,
                not_(commits.c.sha.in_(base_commits_subquery))
            )
        ).order_by(commits.c.date.desc()).limit(limit)
        
        diff_commits = await database.fetch_all(query)
        
        logger.info(f"✅ Found {len(diff_commits)} commits in {head_branch} but not in {base_branch}")
        return diff_commits
        
    except Exception as e:
        logger.error(f"❌ Error comparing branches {base_branch}...{head_branch}: {e}")
        return []

async def validate_and_fix_commit_branch_consistency(repo_id: int):
    """
    Kiểm tra và sửa inconsistency giữa branch_id và branch_name
    """
    try:
        from db.models.branches import branches
        
        # Find commits with inconsistent branch data
        inconsistent_query = select(
            commits.c.id,
            commits.c.sha,
            commits.c.branch_id,
            commits.c.branch_name,
            branches.c.name.label('actual_branch_name')
        ).select_from(
            commits.join(branches, commits.c.branch_id == branches.c.id)
        ).where(
            and_(
                commits.c.repo_id == repo_id,
                commits.c.branch_name != branches.c.name
            )
        )
        
        inconsistent_commits = await database.fetch_all(inconsistent_query)
        
        if inconsistent_commits:
            logger.warning(f"⚠️ Found {len(inconsistent_commits)} commits with inconsistent branch data")
            
            # Fix inconsistencies
            for commit in inconsistent_commits:
                update_query = update(commits).where(
                    commits.c.id == commit.id
                ).values(
                    branch_name=commit.actual_branch_name
                )
                await database.execute(update_query)
                
            logger.info(f"✅ Fixed {len(inconsistent_commits)} commit branch inconsistencies")
        
        return len(inconsistent_commits)
        
    except Exception as e:
        logger.error(f"❌ Error validating commit-branch consistency: {e}")
        return 0