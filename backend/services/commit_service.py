from db.models.commits import commits
from db.database import database
from sqlalchemy import select, insert, update, and_
from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Any
from utils.commit_analyzer import CommitAnalyzer

logger = logging.getLogger(__name__)

def parse_github_datetime(date_str):
    """Convert GitHub API datetime string to Python datetime object (timezone-naive UTC)"""
    if not date_str:
        return None
    
    try:
        # GitHub datetime format: 2021-03-06T14:28:54Z
        if date_str.endswith('Z'):
            date_str = date_str[:-1] + '+00:00'  # Replace Z with +00:00 for proper parsing
        
        # Parse as timezone-aware datetime first
        dt_aware = datetime.fromisoformat(date_str)
        
        # Convert to UTC and make timezone-naive for database storage
        dt_utc = dt_aware.astimezone(timezone.utc)
        return dt_utc.replace(tzinfo=None)
        
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to parse datetime '{date_str}': {e}")
        return None

def normalize_datetime(dt):
    """Normalize datetime to timezone-naive UTC for consistent database storage"""
    if dt is None:
        return None
    
    if dt.tzinfo is not None:
        # Convert timezone-aware to UTC and make timezone-naive
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        # Already timezone-naive, assume it's UTC
        return dt
# lưu commit 
async def save_commit(commit_data, force_update=False):
    """
    Save commit with full data model support including enhanced analysis
    
    Args:
        commit_data: Commit data to save
        force_update: If True, update existing commit with new data
    
    Returns:
        int: Commit ID
    """
    try:
        # Kiểm tra commit đã tồn tại chưa
        query = select(commits).where(commits.c.sha == commit_data["sha"])
        existing_commit = await database.fetch_one(query)

        # Extract enhanced metadata using analyzer
        enhanced_metadata = {}
        if commit_data.get("files") or commit_data.get("stats"):
            enhanced_metadata = CommitAnalyzer.extract_commit_metadata(commit_data)
        
        # Use GitHub API data if available, otherwise use analyzer
        additions = commit_data.get("additions") or commit_data.get("stats", {}).get("additions", 0)
        deletions = commit_data.get("deletions") or commit_data.get("stats", {}).get("deletions", 0)
        total_changes = commit_data.get("total_changes") or (additions + deletions)
        files_changed = commit_data.get("files_changed") or enhanced_metadata.get("files_changed", 0)
        
        # Use GitHub API metadata if available
        modified_files = commit_data.get("modified_files") or enhanced_metadata.get("modified_files", [])
        file_types = commit_data.get("file_types") or enhanced_metadata.get("file_types", {})
        modified_directories = commit_data.get("modified_directories") or enhanced_metadata.get("modified_directories", {})
        is_merge = commit_data.get("is_merge", False)
        
        # Analyze commit message for change type
        message = commit_data.get("message", "")
        change_type = CommitAnalyzer.detect_change_type(message)
        commit_size = CommitAnalyzer.categorize_commit_size(total_changes)

        # Prepare full commit entry with all model fields including enhanced fields
        commit_entry = {
            "sha": commit_data["sha"],
            "message": message,
            "author_name": commit_data.get("author_name", ""),
            "author_email": commit_data.get("author_email", ""),
            "committer_name": commit_data.get("committer_name"),
            "committer_email": commit_data.get("committer_email"),
            "repo_id": commit_data["repo_id"],
            "branch_id": commit_data.get("branch_id"),
            "branch_name": commit_data.get("branch_name"),
            "author_role_at_commit": commit_data.get("author_role_at_commit"),
            "author_permissions_at_commit": commit_data.get("author_permissions_at_commit"),
            "date": normalize_datetime(parse_github_datetime(commit_data.get("date"))),
            "committer_date": normalize_datetime(parse_github_datetime(commit_data.get("committer_date"))),
            "insertions": additions,
            "deletions": deletions,
            "files_changed": files_changed,
            "parent_sha": commit_data.get("parent_sha"),
            "is_merge": is_merge,
            "merge_from_branch": commit_data.get("merge_from_branch"),
            # Enhanced fields from GitHub API
            "modified_files": modified_files,
            "file_types": file_types,
            "modified_directories": modified_directories,
            "total_changes": total_changes,
            "change_type": change_type,
            "commit_size": commit_size,
            # User IDs sẽ được resolve sau nếu có user mapping
            "author_user_id": commit_data.get("author_user_id"),
            "committer_user_id": commit_data.get("committer_user_id"),
            "diff_content": commit_data.get("diff_content"),
        }

        if existing_commit:
            if force_update:
                # Update existing commit with new enhanced metadata
                # Add last_synced timestamp
                commit_entry["last_synced"] = datetime.utcnow()
                
                update_query = update(commits).where(
                    commits.c.sha == commit_data["sha"]
                ).values(commit_entry)
                
                await database.execute(update_query)
                logger.info(f"Updated existing commit: {commit_data['sha']} with enhanced metadata")
                return existing_commit.id
            else:
                logger.info(f"Commit {commit_data['sha']} already exists, skipping")
                return existing_commit.id

        # Chèn commit mới
        query = insert(commits).values(commit_entry)
        result = await database.execute(query)
        logger.info(f"Created new commit: {commit_data['sha']}")
        return result
        
    except Exception as e:
        logger.error(f"Error saving commit {commit_data.get('sha')}: {e}")
        raise e

async def save_multiple_commits(commits_data: list, repo_id: int, branch_name: str = None, branch_id: int = None, force_update: bool = False):
    """
    Batch save multiple commits efficiently
    
    Args:
        commits_data: List of commit data from GitHub API
        repo_id: Repository ID
        branch_name: Branch name (optional)
        branch_id: Branch ID (optional)
        force_update: If True, update existing commits with new data
    
    Returns:
        int: Number of commits saved or updated
    """
    if not commits_data:
        return 0
    
    # Prepare batch data
    new_commits = []
    existing_commits = {}
    update_count = 0
    
    # Get all SHAs from the data
    shas = [commit.get("sha") for commit in commits_data if commit.get("sha")]
    if not shas:
        return 0
    
    # Check existing commits
    query = select(commits).where(commits.c.sha.in_(shas))
    existing_results = await database.fetch_all(query)
    
    # Map existing commits by SHA for quick lookup
    for row in existing_results:
        existing_commits[row.sha] = row
    
    # Process commits data
    for commit_data in commits_data:
        sha = commit_data.get("sha")
        if not sha:
            continue
        
        # Extract commit info from GitHub API response
        commit_info = commit_data.get("commit", {})
        author_info = commit_info.get("author", {})
        committer_info = commit_info.get("committer", {})
        stats = commit_data.get("stats", {})
        
        # Enhanced analysis for each commit
        enhanced_metadata = {}
        if commit_data.get("files") or stats:
            enhanced_metadata = CommitAnalyzer.extract_commit_metadata(commit_data)
        
        # Calculate enhanced statistics
        additions = stats.get("additions", 0)
        deletions = stats.get("deletions", 0)
        total_changes = additions + deletions
        
        # Analyze commit message
        message = commit_info.get("message", "")
        change_type = CommitAnalyzer.detect_change_type(message)
        commit_size = CommitAnalyzer.categorize_commit_size(total_changes)
        
        # Check if this is a merge commit
        parents = commit_data.get("parents", [])
        is_merge = len(parents) > 1
        parent_sha = parents[0].get("sha") if parents else None
        
        commit_entry = {
            "sha": sha,
            "message": message,
            "author_name": author_info.get("name", ""),
            "author_email": author_info.get("email", ""),
            "committer_name": committer_info.get("name"),
            "committer_email": committer_info.get("email"),
            "repo_id": repo_id,
            "branch_id": branch_id,
            "branch_name": branch_name,
            "date": normalize_datetime(parse_github_datetime(author_info.get("date"))),
            "committer_date": normalize_datetime(parse_github_datetime(committer_info.get("date"))),
            "insertions": additions,
            "deletions": deletions,
            "files_changed": enhanced_metadata.get("files_changed") or len(commit_data.get("files", [])) if commit_data.get("files") else None,
            "parent_sha": parent_sha,
            "is_merge": is_merge,
            # Enhanced fields
            "modified_files": enhanced_metadata.get("modified_files"),
            "file_types": enhanced_metadata.get("file_types"),
            "modified_directories": enhanced_metadata.get("modified_directories"),
            "total_changes": total_changes,
            "change_type": change_type,
            "commit_size": commit_size,
            "last_synced": datetime.utcnow(),
            # User IDs và permissions sẽ được resolve sau nếu có user service
        }
        
        if sha in existing_commits:
            if force_update:
                # Update existing commit
                update_query = update(commits).where(
                    commits.c.sha == sha
                ).values(commit_entry)
                await database.execute(update_query)
                update_count += 1
                logger.info(f"Updated existing commit: {sha} with enhanced metadata")
            else:
                logger.debug(f"Commit {sha} already exists, skipping")
        else:
            # Add to new commits batch
            new_commits.append(commit_entry)
    
    # Batch insert new commits
    if new_commits:
        query = commits.insert()
        await database.execute_many(query, new_commits)
        logger.info(f"Batch saved {len(new_commits)} new commits for repo_id {repo_id}")
    
    # Return total number of inserts and updates
    total_processed = len(new_commits) + update_count
    logger.info(f"Total processed: {total_processed} ({len(new_commits)} new, {update_count} updated)")
    return total_processed

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

async def get_enhanced_commit_statistics(repo_id: int, branch_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get enhanced commit statistics including file type distributions and change analysis
    
    Args:
        repo_id: Repository ID
        branch_name: Optional branch name filter
        
    Returns:
        Dictionary with comprehensive commit statistics
    """
    try:
        from sqlalchemy import func, desc
        
        # Base query
        base_query = select(commits).where(commits.c.repo_id == repo_id)
        if branch_name:
            base_query = base_query.where(commits.c.branch_name == branch_name)
        
        # Get all commits for analysis
        all_commits = await database.fetch_all(base_query)
        
        if not all_commits:
            return {
                "total_commits": 0,
                "message": "No commits found"
            }
        
        # Calculate basic statistics
        total_commits = len(all_commits)
        total_additions = sum(c.insertions or 0 for c in all_commits)
        total_deletions = sum(c.deletions or 0 for c in all_commits)
        total_files_changed = sum(c.files_changed or 0 for c in all_commits)
        
        # Aggregate enhanced statistics
        file_type_distribution = {}
        directory_distribution = {}
        commit_size_distribution = {}
        change_type_distribution = {}
        language_distribution = {}
        
        for commit in all_commits:
            # File types
            if commit.file_types:
                for file_type, count in commit.file_types.items():
                    file_type_distribution[file_type] = file_type_distribution.get(file_type, 0) + count
            
            # Directories
            if commit.modified_directories:
                for directory, count in commit.modified_directories.items():
                    directory_distribution[directory] = directory_distribution.get(directory, 0) + count
            
            # Commit sizes
            if commit.commit_size:
                commit_size_distribution[commit.commit_size] = commit_size_distribution.get(commit.commit_size, 0) + 1
            
            # Change types
            if commit.change_type:
                change_type_distribution[commit.change_type] = change_type_distribution.get(commit.change_type, 0) + 1
        
        # Calculate averages
        avg_additions = total_additions / total_commits if total_commits > 0 else 0
        avg_deletions = total_deletions / total_commits if total_commits > 0 else 0
        avg_files_changed = total_files_changed / total_commits if total_commits > 0 else 0
        
        # Get top contributors
        contributor_stats = {}
        for commit in all_commits:
            author = commit.author_name
            if author:
                if author not in contributor_stats:
                    contributor_stats[author] = {
                        "commits": 0,
                        "additions": 0,
                        "deletions": 0,
                        "files_changed": 0
                    }
                contributor_stats[author]["commits"] += 1
                contributor_stats[author]["additions"] += commit.insertions or 0
                contributor_stats[author]["deletions"] += commit.deletions or 0
                contributor_stats[author]["files_changed"] += commit.files_changed or 0
        
        # Sort contributors by commit count
        top_contributors = sorted(
            contributor_stats.items(),
            key=lambda x: x[1]["commits"],
            reverse=True
        )[:10]
        
        return {
            "repository_id": repo_id,
            "branch_name": branch_name,
            "total_commits": total_commits,
            "code_statistics": {
                "total_additions": total_additions,
                "total_deletions": total_deletions,
                "total_files_changed": total_files_changed,
                "average_additions_per_commit": round(avg_additions, 2),
                "average_deletions_per_commit": round(avg_deletions, 2),
                "average_files_changed_per_commit": round(avg_files_changed, 2)
            },
            "distributions": {
                "file_types": dict(sorted(file_type_distribution.items(), key=lambda x: x[1], reverse=True)),
                "directories": dict(sorted(directory_distribution.items(), key=lambda x: x[1], reverse=True)),
                "commit_sizes": commit_size_distribution,
                "change_types": change_type_distribution
            },
            "top_contributors": [
                {
                    "name": name,
                    "stats": stats
                }
                for name, stats in top_contributors
            ],
            "analysis_metadata": {
                "analyzed_at": datetime.utcnow().isoformat(),
                "total_records_analyzed": total_commits
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting enhanced commit statistics: {e}")
        raise e

async def analyze_commit_trends(repo_id: int, days: int = 30) -> Dict[str, Any]:
    """
    Analyze commit trends over time
    
    Args:
        repo_id: Repository ID
        days: Number of days to analyze
        
    Returns:
        Dictionary with trend analysis
    """
    try:
        from sqlalchemy import func
        from datetime import timedelta
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get commits in date range
        query = select(commits).where(
            and_(
                commits.c.repo_id == repo_id,
                commits.c.date >= start_date,
                commits.c.date <= end_date
            )
        ).order_by(commits.c.date.desc())
        
        recent_commits = await database.fetch_all(query)
        
        if not recent_commits:
            return {
                "period_days": days,
                "commits_found": 0,
                "message": "No commits found in the specified period"
            }
        
        # Group by day
        daily_stats = {}
        for commit in recent_commits:
            day_key = commit.date.date().isoformat()
            
            if day_key not in daily_stats:
                daily_stats[day_key] = {
                    "commit_count": 0,
                    "additions": 0,
                    "deletions": 0,
                    "files_changed": 0,
                    "contributors": set()
                }
            
            daily_stats[day_key]["commit_count"] += 1
            daily_stats[day_key]["additions"] += commit.insertions or 0
            daily_stats[day_key]["deletions"] += commit.deletions or 0
            daily_stats[day_key]["files_changed"] += commit.files_changed or 0
            daily_stats[day_key]["contributors"].add(commit.author_name)
        
        # Convert sets to counts and sort by date
        formatted_daily_stats = {}
        for day, stats in daily_stats.items():
            formatted_daily_stats[day] = {
                **stats,
                "unique_contributors": len(stats["contributors"])
            }
            del formatted_daily_stats[day]["contributors"]
        
        # Sort by date
        sorted_daily_stats = dict(sorted(formatted_daily_stats.items()))
        
        # Calculate trends
        total_commits_period = len(recent_commits)
        avg_commits_per_day = total_commits_period / days
        
        return {
            "period_days": days,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_commits": total_commits_period,
                "average_commits_per_day": round(avg_commits_per_day, 2),
                "active_days": len(daily_stats),
                "total_contributors": len(set(c.author_name for c in recent_commits))
            },
            "daily_breakdown": sorted_daily_stats
        }
        
    except Exception as e:
        logger.error(f"Error analyzing commit trends: {e}")
        raise e