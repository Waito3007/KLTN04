# backend/api/routes/commit.py
"""
Commit API Routes - Comprehensive commit management system

ENDPOINT CATEGORIES:
1. DATABASE QUERIES (Fast, stored data):
   - /commits/{owner}/{repo}/branches/{branch_name}/commits - Get commits by branch from DB
   - /commits/{owner}/{repo}/commits - Get all repo commits from DB with filters
   - /commits/{owner}/{repo}/branches - Get all branches with commit stats
   - /commits/{owner}/{repo}/compare/{base}...{head} - Compare commits between branches
   - /commits/{sha} - Get specific commit details

2. GITHUB DIRECT FETCH (Real-time, live data):
   - /github/{owner}/{repo}/branches/{branch_name}/commits - Fetch branch commits from GitHub API
   - /github/{owner}/{repo}/commits - Fetch repo commits from GitHub API with full filters

3. SYNC & MANAGEMENT:
   - /github/{owner}/{repo}/sync-commits - Sync commits from GitHub to database
   - /github/{owner}/{repo}/sync-all-branches-commits - Sync all branches' commits
   - /commits/{owner}/{repo}/validate-commit-consistency - Validate & fix data consistency

4. ANALYTICS & STATS:
   - /github/{owner}/{repo}/commit-stats - Get comprehensive commit statistics

USAGE GUIDELINES:
- Use DATABASE endpoints for fast queries on stored data
- Use GITHUB DIRECT endpoints for real-time, up-to-date data
- Use SYNC endpoints to populate/update database from GitHub
- Use ANALYTICS endpoints for insights and statistics
"""
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, status, Request
from fastapi.responses import JSONResponse
import httpx
import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from services.commit_service import (
    get_commit_by_sha, save_commit, get_commits_by_branch_safe, 
    get_commits_by_repo_id, get_commit_statistics, 
    get_enhanced_commit_statistics, analyze_commit_trends
)
from core.security import security # Import security dependency
from services.repo_service import get_repo_id_by_owner_and_name, get_repository
from services.branch_service import get_branches_by_repo_id
from services.github_service import fetch_commits, fetch_commit_details

commit_router = APIRouter()
logger = logging.getLogger(__name__)

async def github_api_call(url: str, token: str, params: dict = None):
    """Helper function for GitHub API calls with error handling"""
    headers = {
        "Authorization": token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers, params=params or {})
        
        if response.status_code == 429:
            raise HTTPException(status_code=429, detail="GitHub API rate limit exceeded")
        elif response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"GitHub API error: {response.text}"
            )
        
        return response.json()

async def fetch_raw_github_content(url: str, token: str):
    """Helper function to fetch raw content (e.g., diffs) from GitHub API."""
    headers = {
        "Authorization": token,
        "Accept": "application/vnd.github.v3.diff", # Yêu cầu định dạng diff
        "X-GitHub-Api-Version": "2022-11-28"
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url, headers=headers)
        
        if response.status_code == 429:
            raise HTTPException(status_code=429, detail="GitHub API rate limit exceeded")
        elif response.status_code != 200:
            logger.error(f"GitHub API error fetching raw content from {url}. Status: {response.status_code}, Detail: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"GitHub API error fetching raw content: {response.text}"
            )
        
        return response.text # Trả về văn bản thuần túy

# ==================== NEW BRANCH-SPECIFIC COMMIT ENDPOINTS ====================

@commit_router.get("/commits/{owner}/{repo}/branches/{branch_name}/commits")
async def get_branch_commits(
    owner: str,
    repo: str,
    branch_name: str,
    limit: int = Query(50, ge=1, le=500, description="Number of commits to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    request: Request = None
):
    """
    Lấy commits của một branch cụ thể với validation đầy đủ
    """
    try:
        from services.commit_service import get_repo_id_by_owner_and_name, get_commits_by_branch_safe, get_commits_by_branch_fallback
        
        # Get repo_id
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo} not found")
        
        # Try safe method first (with branch_id validation)
        commits_data = await get_commits_by_branch_safe(repo_id, branch_name, limit, offset)
        
        # Fallback to branch_name only if safe method returns empty
        if not commits_data:
            logger.warning(f"Safe method returned empty, trying fallback for {owner}/{repo}:{branch_name}")
            commits_data = await get_commits_by_branch_fallback(repo_id, branch_name, limit, offset)
        
        # Convert to dict format for JSON response
        commits_list = []
        for commit in commits_data:
            commit_dict = {
                "id": commit.id,
                "sha": commit.sha,
                "message": commit.message,
                "author_name": commit.author_name,
                "author_email": commit.author_email,
                "committer_name": commit.committer_name,
                "committer_email": commit.committer_email,
                "date": commit.date.isoformat() if commit.date else None,
                "committer_date": commit.committer_date.isoformat() if commit.committer_date else None,
                "insertions": commit.insertions,
                "deletions": commit.deletions,
                "files_changed": commit.files_changed,
                "modified_files": commit.modified_files,
                "is_merge": commit.is_merge,
                "merge_from_branch": commit.merge_from_branch,
                "branch_name": commit.branch_name,
                "author_role_at_commit": commit.author_role_at_commit,
                "diff_content": commit.diff_content
            }
            commits_list.append(commit_dict)
        
        return {
            "repository": f"{owner}/{repo}",
            "branch": branch_name,
            "commits": commits_list,
            "count": len(commits_list),
            "limit": limit,
            "offset": offset,
            "total_found": len(commits_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting commits for branch {branch_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@commit_router.get("/commits/{owner}/{repo}/branches")
async def get_repository_branches_with_commits(
    owner: str,
    repo: str,
    request: Request = None
):
    """
    Lấy danh sách tất cả branches với thống kê commits
    """
    try:
        from services.commit_service import get_repo_id_by_owner_and_name, get_all_branches_with_commit_stats
        
        # Get repo_id
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo} not found")
        
        # Get branches with commit stats
        branches_data = await get_all_branches_with_commit_stats(repo_id)
        
        # Format response
        branches_list = []
        for branch in branches_data:
            branch_dict = {
                "id": branch.id,
                "name": branch.name,
                "is_default": branch.is_default,
                "is_protected": branch.is_protected,
                "stored_commit_count": branch.commits_count,
                "actual_commit_count": branch.actual_commit_count,
                "latest_commit_date": branch.latest_commit_date.isoformat() if branch.latest_commit_date else None,
                "last_synced_commit_date": branch.last_commit_date.isoformat() if branch.last_commit_date else None
            }
            branches_list.append(branch_dict)
        
        return {
            "repository": f"{owner}/{repo}",
            "branches": branches_list,
            "total_branches": len(branches_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting branches for repo {owner}/{repo}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@commit_router.get("/commits/{owner}/{repo}/compare/{base_branch}...{head_branch}")
async def compare_branch_commits(
    owner: str,
    repo: str,
    base_branch: str,
    head_branch: str,
    limit: int = Query(100, ge=1, le=500, description="Number of commits to return"),
    request: Request = None
):
    """
    So sánh commits giữa 2 branches (commits in head_branch but not in base_branch)
    """
    try:
        from services.commit_service import get_repo_id_by_owner_and_name, compare_commits_between_branches
        
        # Get repo_id
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo} not found")
        
        # Get diff commits
        diff_commits = await compare_commits_between_branches(repo_id, base_branch, head_branch, limit)
        
        # Format response
        commits_list = []
        for commit in diff_commits:
            commit_dict = {
                "sha": commit.sha,
                "message": commit.message,
                "author_name": commit.author_name,
                "author_email": commit.author_email,
                "date": commit.date.isoformat() if commit.date else None,
                "insertions": commit.insertions,
                "deletions": commit.deletions,
                "files_changed": commit.files_changed,
                "is_merge": commit.is_merge,
                "diff_content": commit.diff_content
            }
            commits_list.append(commit_dict)
        
        return {
            "repository": f"{owner}/{repo}",
            "comparison": f"{base_branch}...{head_branch}",
            "commits_ahead": commits_list,
            "commits_ahead_count": len(commits_list),
            "base_branch": base_branch,
            "head_branch": head_branch
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing branches {base_branch}...{head_branch}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@commit_router.post("/commits/{owner}/{repo}/validate-commit-consistency")
async def validate_commit_branch_consistency(
    owner: str,
    repo: str,
    request: Request = None
):
    """
    Kiểm tra và sửa inconsistency giữa branch_id và branch_name trong commits
    """
    try:
        from services.commit_service import get_repo_id_by_owner_and_name, validate_and_fix_commit_branch_consistency
        
        # Get repo_id
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo} not found")
        
        # Validate and fix consistency
        fixed_count = await validate_and_fix_commit_branch_consistency(repo_id)
        
        return {
            "repository": f"{owner}/{repo}",
            "message": "Commit-branch consistency validation completed",
            "inconsistencies_fixed": fixed_count,
            "status": "success" if fixed_count >= 0 else "error"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating commit consistency for {owner}/{repo}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Endpoint đồng bộ commit từ GitHub về database - Optimized version
@commit_router.post("/github/{owner}/{repo}/sync-commits")
async def sync_commits(
    owner: str,
    repo: str,
    request: Request,
    branch: str = Query("main", description="Branch name to sync commits from"),
    since: Optional[str] = Query(None, description="Only commits after this date (ISO format)"),
    until: Optional[str] = Query(None, description="Only commits before this date (ISO format)"),
    per_page: int = Query(100, ge=1, le=100, description="Number of commits per page"),
    max_pages: int = Query(10, ge=1, le=50, description="Maximum pages to fetch"),
    include_stats: bool = Query(False, description="Include commit statistics (slower)"),
    force_update: bool = Query(True, description="Force update existing commits with new data")
):
    """
    Đồng bộ commits từ GitHub với full model support
    
    Hỗ trợ tất cả các fields trong commit model:
    - Basic info (sha, message, author, committer, dates)
    - Statistics (insertions, deletions, files_changed)
    - Relationships (repo_id, branch_id, parent_sha)
    - Metadata (is_merge, merge_from_branch)
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid GitHub token")
    
    try:
        logger.info(f"Starting commit sync for {owner}/{repo}:{branch}")
        
        # 1. Validate repository exists
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository not found in database")
        
        # 2. Get branch info
        branches = await get_branches_by_repo_id(repo_id)
        branch_id = None
        for b in branches:
            if b['name'] == branch:
                branch_id = b['id']
                break
        
        if not branch_id:
            logger.warning(f"Branch {branch} not found in database, using branch_name only")
        
        # 3. Fetch commits with enhanced data
        all_commits = []
        page = 1
        
        while page <= max_pages:
            logger.info(f"Fetching commits page {page}/{max_pages}")
            
            params = {
                "sha": branch,
                "per_page": per_page,
                "page": page
            }
            
            if since:
                params["since"] = since
            if until:
                params["until"] = until
                
            url = f"https://api.github.com/repos/{owner}/{repo}/commits"
            commits_data = await github_api_call(url, token, params)
            
            if not commits_data:
                break
            
            # Optionally enhance commits with detailed stats
            if include_stats:
                logger.info(f"Enhancing {len(commits_data)} commits with detailed stats...")
                for commit in commits_data:
                    sha = commit.get("sha")
                    if sha:
                        try:
                            # Fetch detailed commit info including stats
                            detail_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
                            detailed_commit = await github_api_call(detail_url, token)
                            
                            # Merge detailed info
                            if detailed_commit:
                                commit["detailed_stats"] = detailed_commit.get("stats", {})
                                commit["files"] = detailed_commit.get("files", [])
                                
                        except Exception as e:
                            logger.warning(f"Could not fetch details for commit {sha}: {e}")
                        
                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.05)
            
            all_commits.extend(commits_data)
            
            if len(commits_data) < per_page:
                break
                
            page += 1
            await asyncio.sleep(0.1)
        
        logger.info(f"Fetched {len(all_commits)} commits from GitHub")
        
        # 4. Process and save commits with full model data
        saved_count = await save_multiple_commits(
            commits_data=all_commits,
            repo_id=repo_id,
            branch_name=branch,
            branch_id=branch_id,
            force_update=force_update
        )
        
        return {
            "message": f"Synced {saved_count} commits from GitHub",
            "repository": f"{owner}/{repo}",
            "branch": branch,
            "commits_fetched": len(all_commits),
            "commits_processed": saved_count,
            "force_update": force_update
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing commits for {owner}/{repo}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Commit sync failed: {str(e)}")

# Endpoint đồng bộ commits cho tất cả branches
@commit_router.post("/github/{owner}/{repo}/sync-all-branches-commits")
async def sync_all_branches_commits(
    owner: str,
    repo: str,
    request: Request,
    since: Optional[str] = Query(None, description="Only commits after this date"),
    per_page: int = Query(50, ge=1, le=100),
    max_pages_per_branch: int = Query(5, ge=1, le=20),
    force_update: bool = Query(True, description="Force update existing commits with new data")
):
    """
    Đồng bộ commits cho tất cả branches của repository
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid GitHub token")
    
    try:
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        # Get all branches
        branches = await get_branches_by_repo_id(repo_id)
        
        if not branches:
            raise HTTPException(status_code=404, detail="No branches found for repository")
        
        total_saved = 0
        branch_results = []
        
        for branch in branches:
            branch_name = branch['name']
            logger.info(f"Processing commits for branch: {branch_name}")
            
            try:
                # Fetch commits for this branch
                params = {
                    "sha": branch_name,
                    "per_page": per_page
                }
                if since:
                    params["since"] = since
                
                all_commits = []
                page = 1
                
                while page <= max_pages_per_branch:
                    params["page"] = page
                    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
                    commits_data = await github_api_call(url, token, params)
                    
                    if not commits_data:
                        break
                    
                    all_commits.extend(commits_data)
                    
                    if len(commits_data) < per_page:
                        break
                    
                    page += 1
                    await asyncio.sleep(0.1)
                
                # Save commits for this branch
                saved_count = await save_multiple_commits(
                    commits_data=all_commits,
                    repo_id=repo_id,
                    branch_name=branch_name,
                    branch_id=branch['id'],
                    force_update=force_update
                )
                
                total_saved += saved_count
                branch_results.append({
                    "branch": branch_name,
                    "commits_fetched": len(all_commits),
                    "commits_processed": saved_count,
                    "force_update": force_update
                })
                
            except Exception as e:
                logger.warning(f"Failed to sync commits for branch {branch_name}: {e}")
                branch_results.append({
                    "branch": branch_name,
                    "error": str(e)
                })
        
        return {
            "message": f"Synced commits for {len(branches)} branches",
            "repository": f"{owner}/{repo}",
            "total_commits_processed": total_saved,
            "branch_results": branch_results,
            "force_update": force_update
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-branch sync failed: {str(e)}")

# REDUNDANT ENDPOINT REMOVED - Use /commits/{owner}/{repo}/branches/{branch_name}/commits instead

# Endpoint lấy commit chi tiết
@commit_router.get("/commits/{sha}")
async def get_commit_details(sha: str):
    """Get detailed information about a specific commit"""
    try:
        commit = await get_commit_by_sha(sha)
        if not commit:
            raise HTTPException(status_code=404, detail="Commit not found")
        
        return commit
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get commit: {str(e)}")

# Endpoint thống kê commits
@commit_router.get("/github/{owner}/{repo}/commit-stats")
async def get_repository_commit_statistics(owner: str, repo: str):
    """Get comprehensive commit statistics for a repository"""
    try:
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        stats = await get_commit_statistics(repo_id)
        
        return {
            "repository": f"{owner}/{repo}",
            "statistics": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get commit statistics: {str(e)}")

# CONSOLIDATED ENDPOINT - This replaces the removed /github/{owner}/{repo}/commits
@commit_router.get("/commits/{owner}/{repo}/commits")
async def get_repository_commits_from_database(
    owner: str,
    repo: str,
    branch: Optional[str] = Query(None, description="Filter commits by branch name (redirects to branch-specific endpoint)"),
    limit: int = Query(50, ge=1, le=500, description="Number of commits to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    since: Optional[str] = Query(None, description="Only commits after this date (ISO format)"),
    until: Optional[str] = Query(None, description="Only commits before this date (ISO format)"),
    request: Request = None
):
    """
    Lấy commits của repository từ database với filtering nâng cao
    Note: Nếu chỉ định branch, sẽ redirect đến endpoint branch-specific cho hiệu suất tốt hơn
    """
    try:
        # If branch is specified, redirect to branch-specific endpoint
        if branch:
            return await get_branch_commits(owner, repo, branch, limit, offset, request)
        
        # Otherwise, get all commits (existing logic)
        from services.commit_service import get_repo_id_by_owner_and_name
        from db.models.commits import commits
        from db.database import database
        from sqlalchemy import select, and_
        
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo} not found")
        
        # Build query with filters
        query = select(commits).where(commits.c.repo_id == repo_id)
        
        # Add date filters if provided
        if since:
            try:
                since_date = datetime.fromisoformat(since.replace('Z', '+00:00'))
                query = query.where(commits.c.date >= since_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid 'since' date format")
        
        if until:
            try:
                until_date = datetime.fromisoformat(until.replace('Z', '+00:00'))
                query = query.where(commits.c.date <= until_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid 'until' date format")
        
        # Apply pagination and ordering
        query = query.order_by(commits.c.date.desc()).limit(limit).offset(offset)
        
        commits_data = await database.fetch_all(query)
        
        # Format response
        commits_list = []
        for commit in commits_data:
            commit_dict = {
                "id": commit.id,
                "sha": commit.sha,
                "message": commit.message,
                "author_name": commit.author_name,
                "author_email": commit.author_email,
                "date": commit.date.isoformat() if commit.date else None,
                "branch_name": commit.branch_name,
                "insertions": commit.insertions,
                "deletions": commit.deletions,
                "files_changed": commit.files_changed,
                "is_merge": commit.is_merge,
                "diff_content": commit.diff_content
            }
            commits_list.append(commit_dict)
        
        return {
            "repository": f"{owner}/{repo}",
            "commits": commits_list,
            "count": len(commits_list),
            "limit": limit,
            "offset": offset,
            "filters": {
                "since": since,
                "until": until,
                "branch": branch
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting commits for repo {owner}/{repo}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ==================== GITHUB DIRECT FETCH ENDPOINTS ====================

@commit_router.get("/github/{owner}/{repo}/branches/{branch_name}/commits")
async def get_branch_commits_from_github(
    owner: str,
    repo: str,
    branch_name: str,
    request: Request,
    per_page: int = Query(30, ge=1, le=100, description="Number of commits per page"),
    page: int = Query(1, ge=1, le=100, description="Page number"),
    since: Optional[str] = Query(None, description="Only commits after this date (ISO format)"),
    until: Optional[str] = Query(None, description="Only commits before this date (ISO format)")
):
    """
    Fetch commits directly from GitHub for a specific branch (real-time data)
    
    This endpoint fetches commits directly from GitHub API without storing in database.
    Use this when you need real-time, up-to-date commit data.
    For faster access to stored data, use /commits/{owner}/{repo}/branches/{branch_name}/commits instead.
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid GitHub token")
    
    try:
        logger.info(f"Fetching commits from GitHub for {owner}/{repo}:{branch_name}")
        
        # Build parameters for GitHub API
        params = {
            "sha": branch_name,
            "per_page": per_page,
            "page": page
        }
        
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        
        # Fetch commits from GitHub
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        commits_data = await github_api_call(url, token, params)
        
        if not commits_data:
            return {
                "repository": f"{owner}/{repo}",
                "branch": branch_name,
                "commits": [],
                "count": 0,
                "page": page,
                "per_page": per_page,
                "source": "github_api",
                "message": "No commits found"
            }
        
        # Format commits to match our standard response format
        formatted_commits = []
        for commit in commits_data:
            commit_info = commit.get("commit", {})
            author_info = commit_info.get("author", {})
            committer_info = commit_info.get("committer", {})
            
            formatted_commit = {
                "sha": commit.get("sha"),
                "message": commit_info.get("message"),
                "author_name": author_info.get("name"),
                "author_email": author_info.get("email"),
                "author_date": author_info.get("date"),
                "committer_name": committer_info.get("name"),
                "committer_email": committer_info.get("email"),
                "committer_date": committer_info.get("date"),
                "url": commit.get("html_url"),
                "api_url": commit.get("url"),
                "comment_count": commit_info.get("comment_count", 0),
                "verification": commit_info.get("verification", {}),
                "author": commit.get("author"),  # GitHub user info
                "committer": commit.get("committer"),  # GitHub user info
                "parents": [{"sha": p.get("sha"), "url": p.get("url")} for p in commit.get("parents", [])]
            }
            formatted_commits.append(formatted_commit)
        
        return {
            "repository": f"{owner}/{repo}",
            "branch": branch_name,
            "commits": formatted_commits,
            "count": len(formatted_commits),
            "page": page,
            "per_page": per_page,
            "source": "github_api",
            "filters": {
                "since": since,
                "until": until
            },
            "note": "This data is fetched directly from GitHub API. For stored data, use /commits/{owner}/{repo}/branches/{branch_name}/commits"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching commits from GitHub for {owner}/{repo}:{branch_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch commits from GitHub: {str(e)}")

@commit_router.get("/github/{owner}/{repo}/commits")
async def get_repository_commits_from_github(
    owner: str,
    repo: str,
    request: Request,
    sha: Optional[str] = Query(None, description="SHA or branch to start listing commits from"),
    path: Optional[str] = Query(None, description="Only commits containing this file path will be returned"),
    author: Optional[str] = Query(None, description="GitHub username or email address"),
    committer: Optional[str] = Query(None, description="GitHub username or email address"),
    since: Optional[str] = Query(None, description="Only commits after this date (ISO format)"),
    until: Optional[str] = Query(None, description="Only commits before this date (ISO format)"),
    per_page: int = Query(30, ge=1, le=100, description="Number of commits per page"),
    page: int = Query(1, ge=1, le=100, description="Page number")
):
    """
    Fetch commits directly from GitHub for a repository (real-time data)
    
    This endpoint provides comprehensive filtering options available in GitHub API.
    For branch-specific queries, consider using /github/{owner}/{repo}/branches/{branch_name}/commits.
    For stored data, use /commits/{owner}/{repo}/commits.
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid GitHub token")
    
    try:
        logger.info(f"Fetching commits from GitHub for {owner}/{repo}")
        
        # Build parameters for GitHub API with all available filters
        params = {
            "per_page": per_page,
            "page": page
        }
        
        # Add optional filters
        if sha:
            params["sha"] = sha
        if path:
            params["path"] = path
        if author:
            params["author"] = author
        if committer:
            params["committer"] = committer
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        
        # Fetch commits from GitHub
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        commits_data = await github_api_call(url, token, params)
        
        if not commits_data:
            return {
                "repository": f"{owner}/{repo}",
                "commits": [],
                "count": 0,
                "page": page,
                "per_page": per_page,
                "source": "github_api",
                "message": "No commits found"
            }
        
        # Format commits (same as branch-specific endpoint)
        formatted_commits = []
        for commit in commits_data:
            commit_info = commit.get("commit", {})
            author_info = commit_info.get("author", {})
            committer_info = commit_info.get("committer", {})
            
            formatted_commit = {
                "sha": commit.get("sha"),
                "message": commit_info.get("message"),
                "author_name": author_info.get("name"),
                "author_email": author_info.get("email"),
                "author_date": author_info.get("date"),
                "committer_name": committer_info.get("name"),
                "committer_email": committer_info.get("email"),
                "committer_date": committer_info.get("date"),
                "url": commit.get("html_url"),
                "api_url": commit.get("url"),
                "comment_count": commit_info.get("comment_count", 0),
                "verification": commit_info.get("verification", {}),
                "author": commit.get("author"),
                "committer": commit.get("committer"),
                "parents": [{"sha": p.get("sha"), "url": p.get("url")} for p in commit.get("parents", [])]
            }
            formatted_commits.append(formatted_commit)
        
        return {
            "repository": f"{owner}/{repo}",
            "commits": formatted_commits,
            "count": len(formatted_commits),
            "page": page,
            "per_page": per_page,
            "source": "github_api",
            "filters": {
                "sha": sha,
                "path": path,
                "author": author,
                "committer": committer,
                "since": since,
                "until": until
            },
            "note": "This data is fetched directly from GitHub API. For stored data, use /commits/{owner}/{repo}/commits"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching commits from GitHub for {owner}/{repo}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch commits from GitHub: {str(e)}")

# ==================== END GITHUB DIRECT FETCH ENDPOINTS ====================

# ==================== BRANCH-SPECIFIC SYNC ENDPOINT ====================

@commit_router.post("/github/{owner}/{repo}/branches/{branch_name}/sync-commits")
async def sync_branch_commits_enhanced(
    owner: str,
    repo: str,
    branch_name: str,
    request: Request,
    force_refresh: bool = Query(False, description="Force refresh all commits even if they exist"),
    per_page: int = Query(100, ge=1, le=100, description="Number of commits per page"),
    max_pages: int = Query(10, ge=1, le=50, description="Maximum pages to fetch"),
    include_stats: bool = Query(True, description="Include detailed commit statistics"),
    include_diff: bool = Query(False, description="Include commit diff content (slower)"),
    force_update: bool = Query(True, description="Force update existing commits with new data")
):
    """
    Đồng bộ commits cho một branch cụ thể với enhanced features
    
    Endpoint này được thiết kế đặc biệt cho BranchSelector component:
    - Sync commits cho branch được chọn
    - Hỗ trợ force refresh để cập nhật dữ liệu
    - Bao gồm thống kê chi tiết cho commit analysis
    - Tối ưu hóa cho UI responsiveness
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid GitHub token")
    
    try:
        logger.info(f"Starting enhanced commit sync for {owner}/{repo}:{branch_name}")
        
        # 1. Validate repository exists
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository not found in database")
        
        # 2. Get or create branch info
        from services.branch_service import get_branches_by_repo_id, save_branch
        branches = await get_branches_by_repo_id(repo_id)
        branch_id = None
        
        for branch in branches:
            if branch['name'] == branch_name:
                branch_id = branch['id']
                break
        
        # Create branch if not exists
        if not branch_id:
            logger.info(f"Branch {branch_name} not found, creating new branch record")
            try:
                # Fetch branch info from GitHub
                branch_url = f"https://api.github.com/repos/{owner}/{repo}/branches/{branch_name}"
                branch_data = await github_api_call(branch_url, token)
                
                new_branch = {
                    "name": branch_name,
                    "repo_id": repo_id,
                    "sha": branch_data.get("commit", {}).get("sha"),
                    "is_protected": branch_data.get("protected", False)
                }
                await save_branch(new_branch)
                
                # Re-fetch to get branch_id
                branches = await get_branches_by_repo_id(repo_id)
                for branch in branches:
                    if branch['name'] == branch_name:
                        branch_id = branch['id']
                        break
                        
            except Exception as e:
                logger.warning(f"Could not create branch record: {e}")
        
        # 3. Check existing commits if not force refresh
        existing_count = 0
        if not force_refresh:
            from services.commit_service import get_commits_by_branch_safe
            existing_commits = await get_commits_by_branch_safe(repo_id, branch_name, 1, 0)
            existing_count = len(existing_commits) if existing_commits else 0
        
        # 4. Fetch commits from GitHub with enhanced data
        all_commits = []
        page = 1
        
        while page <= max_pages:
            logger.info(f"Fetching commits page {page}/{max_pages} for branch {branch_name}")
            
            params = {
                "sha": branch_name,
                "per_page": per_page,
                "page": page
            }
            
            url = f"https://api.github.com/repos/{owner}/{repo}/commits"
            commits_data = await github_api_call(url, token, params)
            
            if not commits_data:
                break
            
            # Enhance commits with detailed stats if requested
            if include_stats or include_diff:
                logger.info(f"Enhancing {len(commits_data)} commits with detailed stats/diff...")
                for commit in commits_data:
                    sha = commit.get("sha")
                    if sha:
                        try:
                            # Fetch detailed commit info including stats
                            detail_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
                            detailed_commit = await github_api_call(detail_url, token)
                            
                            if detailed_commit:
                                if include_stats:
                                    commit["stats"] = detailed_commit.get("stats", {})
                                    commit["files"] = detailed_commit.get("files", [])
                                
                                if include_diff:
                                    full_diff_content = None
                                    # Ưu tiên: Lấy diff bằng GitHub Compare API (so sánh với commit cha)
                                    parents = commit.get("parents")
                                    parent_sha = parents[0].get("sha") if parents and len(parents) > 0 else None

                                    if parent_sha:
                                        compare_url = f"https://api.github.com/repos/{owner}/{repo}/compare/{parent_sha}...{sha}"
                                        try:
                                            compare_response = await github_api_call(compare_url, token)
                                            # Compare API trả về JSON, trường 'patch' chứa diff
                                            if compare_response and compare_response.get("files"):
                                                compare_diff_lines = []
                                                for file_data in compare_response["files"]:
                                                    if file_data.get("patch"):
                                                        compare_diff_lines.append(file_data["patch"])
                                                full_diff_content = "\n".join(compare_diff_lines)
                                                logger.debug(f"Successfully fetched diff via Compare API for {sha}. Length: {len(full_diff_content) if full_diff_content else 0}")
                                            else:
                                                logger.debug(f"Compare API returned no files or patch for {sha}")
                                        except Exception as compare_e:
                                            logger.warning(f"Could not fetch diff via Compare API for {sha}: {compare_e}. Falling back to detailed_commit patch.")
                                    else:
                                        logger.debug(f"Commit {sha} has no parent. Skipping Compare API call.")

                                    # Dự phòng: Nếu Compare API không thành công hoặc không có diff, sử dụng patch từ detailed_commit
                                    if not full_diff_content and detailed_commit and detailed_commit.get("files"):
                                        patch_content_lines = []
                                        for file_diff in detailed_commit["files"]:
                                            if file_diff.get("patch"):
                                                patch_content_lines.append(file_diff["patch"])
                                        full_diff_content = "\n".join(patch_content_lines)
                                        logger.debug(f"Used fallback patch content from detailed_commit for {sha}. Length: {len(full_diff_content) if full_diff_content else 0}")
                                        logger.debug(f"Detailed commit files for fallback: {[(f.get('filename'), len(f.get('patch', ''))) for f in detailed_commit.get('files', [])]}")
                                    elif not full_diff_content:
                                        logger.debug(f"No diff content available for {sha} after all attempts.")

                                    if full_diff_content:
                                        # Lọc bỏ các dòng tiêu đề Git không mong muốn (diff --git, index, ---, +++)
                                        filtered_diff_lines = []
                                        for line in full_diff_content.splitlines():
                                            if not (line.startswith("diff --git") or \
                                                    line.startswith("index ") or \
                                                    line.startswith("--- a/") or \
                                                    line.startswith("+++ b/")):
                                                filtered_diff_lines.append(line)
                                        commit["diff_content"] = "\n".join(filtered_diff_lines)
                                    else:
                                        commit["diff_content"] = ""
                                
                        except Exception as e:
                            logger.warning(f"Could not fetch details for commit {sha}: {e}")
                        
                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.05)
            
            all_commits.extend(commits_data)
            
            if len(commits_data) < per_page:
                break
                
            page += 1
            await asyncio.sleep(0.1)
        
        logger.info(f"Fetched {len(all_commits)} commits from GitHub for branch {branch_name}")
        
        # 5. Process and save commits with full model data
        saved_count = await save_multiple_commits(
            commits_data=all_commits,
            repo_id=repo_id,
            branch_name=branch_name,
            branch_id=branch_id,
            force_update=force_update
        )
        
        logger.info(f"Successfully processed {saved_count} commits for branch {branch_name}")
        
        # 6. Get final stats
        total_commits_in_db = existing_count + (saved_count if not force_update else len(all_commits))
        
        return {
            "success": True,
            "message": f"Successfully synced commits for branch '{branch_name}'",
            "repository": f"{owner}/{repo}",
            "branch": branch_name,
            "branch_id": branch_id,
            "stats": {
                "total_fetched_from_github": len(all_commits),
                "commits_processed": saved_count,
                "existing_commits_before_sync": existing_count,
                "total_commits_in_database": total_commits_in_db,
                "pages_processed": min(page, max_pages),
                "enhanced_with_stats": include_stats,
                "force_refresh_enabled": force_refresh,
                "force_update_enabled": force_update
            },
            "next_actions": {
                "view_commits": f"/api/commits/{owner}/{repo}/branches/{branch_name}/commits",
                "analyze_commits": f"/api/ai/analyze-repo/{owner}/{repo}?branch={branch_name}"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing commits for branch {branch_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Branch commit sync failed: {str(e)}")

# ==================== END BRANCH-SPECIFIC SYNC ENDPOINT ====================

# Enhanced commit analysis endpoints
@commit_router.get("/commits/{owner}/{repo}/statistics/enhanced")
async def get_enhanced_statistics(
    owner: str,
    repo: str,
    branch_name: Optional[str] = Query(None, description="Branch name filter"),
    request: Request = None
):
    """
    Get enhanced commit statistics with detailed analysis
    """
    try:
        from services.commit_service import get_enhanced_commit_statistics, get_repo_id_by_owner_and_name
        
        # Get repository ID
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo} not found")
        
        # Get enhanced statistics
        stats = await get_enhanced_commit_statistics(repo_id, branch_name)
        
        return {
            "success": True,
            "repository": f"{owner}/{repo}",
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting enhanced statistics for {owner}/{repo}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@commit_router.get("/commits/{owner}/{repo}/trends")
async def analyze_commit_trends_endpoint(
    owner: str,
    repo: str,
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    request: Request = None
):
    """
    Analyze commit trends over specified time period
    """
    try:
        from services.commit_service import analyze_commit_trends, get_repo_id_by_owner_and_name
        
        # Get repository ID
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo} not found")
        
        # Analyze trends
        trends = await analyze_commit_trends(repo_id, days)
        
        return {
            "success": True,
            "repository": f"{owner}/{repo}",
            "trends": trends
        }
        
    except Exception as e:
        logger.error(f"Error analyzing trends for {owner}/{repo}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze trends: {str(e)}")

@commit_router.post("/commits/{owner}/{repo}/analyze")
async def analyze_single_commit(
    owner: str,
    repo: str,
    sha: str = Query(..., description="Commit SHA to analyze"),
    request: Request = None
):
    """
    Analyze a single commit in detail
    """
    try:
        from services.commit_service import get_commit_by_sha
        from utils.commit_analyzer import CommitAnalyzer
        
        # Get commit from database
        commit = await get_commit_by_sha(sha)
        if not commit:
            raise HTTPException(status_code=404, detail=f"Commit {sha} not found")
        
        # Additional analysis
        pattern_analysis = CommitAnalyzer.analyze_commit_pattern(commit.message)
        
        # Compile recommendations
        recommendations = []
        
        if commit.commit_size == "massive":
            recommendations.append("Consider breaking down large commits into smaller, focused changes")
        
        if not pattern_analysis["has_conventional_format"]:
            recommendations.append("Consider using conventional commit format (feat:, fix:, etc.)")
        
        if commit.files_changed and commit.files_changed > 20:
            recommendations.append("This commit touches many files - ensure changes are related")
        
        if pattern_analysis["urgency"] == "high":
            recommendations.append("High priority commit - ensure thorough testing")
        
        return {
            "success": True,
            "commit": {
                "sha": commit.sha,
                "message": commit.message,
                "author": commit.author_name,
                "date": commit.date.isoformat() if commit.date else None,
                "statistics": {
                    "insertions": commit.insertions,
                    "deletions": commit.deletions,
                    "files_changed": commit.files_changed,
                    "total_changes": commit.total_changes,
                    "commit_size": commit.commit_size,
                    "change_type": commit.change_type
                },
                "file_analysis": {
                    "modified_files": commit.modified_files,
                    "file_types": commit.file_types,
                    "modified_directories": commit.modified_directories
                },
                "diff_content": commit.diff_content
            },
            "analysis": {
                "pattern_analysis": pattern_analysis,
                "commit_quality_score": _calculate_commit_quality_score(commit, pattern_analysis)
            },
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing commit {sha}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze commit: {str(e)}")

def _calculate_commit_quality_score(commit, pattern_analysis) -> Dict[str, Any]:
    """
    Calculate a quality score for a commit based on various factors
    """
    score = 100
    factors = []
    
    # Deduct points for issues
    if commit.commit_size == "massive":
        score -= 20
        factors.append("Large commit size (-20)")
    elif commit.commit_size == "large":
        score -= 10
        factors.append("Large commit size (-10)")
    
    if not pattern_analysis["has_conventional_format"]:
        score -= 15
        factors.append("No conventional commit format (-15)")
    
    if commit.files_changed and commit.files_changed > 20:
        score -= 15
        factors.append("Too many files changed (-15)")
    
    # Add points for good practices
    if pattern_analysis["has_conventional_format"]:
        score += 10
        factors.append("Conventional commit format (+10)")
    
    if commit.commit_size == "small":
        score += 5
        factors.append("Good commit size (+5)")
    
    if pattern_analysis["scope"]:
        score += 5
        factors.append("Has scope definition (+5)")
    
    # Ensure score is between 0 and 100
    score = max(0, min(100, score))
    
    return {
        "score": score,
        "grade": _get_grade(score),
        "factors": factors
    }

def _get_grade(score: int) -> str:
    """Convert numeric score to letter grade"""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

# Enhanced GitHub data fetching endpoints
@commit_router.get("/github/{owner}/{repo}/commits/enhanced")
async def fetch_enhanced_commits_from_github(
    owner: str,
    repo: str,
    branch: str = Query("main", description="Branch name"),
    since: Optional[str] = Query(None, description="ISO datetime string"),
    until: Optional[str] = Query(None, description="ISO datetime string"),
    max_commits: int = Query(50, ge=1, le=100, description="Maximum commits to fetch"),
    request: Request = None
):
    """
    Fetch commits with enhanced metadata directly from GitHub API
    Including: files_changed, additions, deletions, total_changes, is_merge, 
              modified_files, file_types, modified_directories
    """
    try:
        from services.github_service import fetch_enhanced_commits_batch
        from core.security import get_current_user_optional
        
        # Get user token if available
        user = await get_current_user_optional(request)
        token = None
        if user and hasattr(request, 'headers'):
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith(("Bearer ", "token ")):
                token = auth_header.split(" ", 1)[1]
        
        # Fetch enhanced commits from GitHub
        enhanced_commits = await fetch_enhanced_commits_batch(
            owner=owner,
            repo=repo,
            branch=branch,
            token=token,
            since=since,
            until=until,
            max_commits=max_commits
        )
        
        return {
            "success": True,
            "repository": f"{owner}/{repo}",
            "branch": branch,
            "commits_count": len(enhanced_commits),
            "commits": enhanced_commits,
            "metadata": {
                "fetched_at": datetime.utcnow().isoformat(),
                "source": "github_api",
                "enhanced_fields": [
                    "files_changed", "additions", "deletions", "total_changes",
                    "is_merge", "modified_files", "file_types", "modified_directories"
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching enhanced commits from GitHub for {owner}/{repo}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch enhanced commits: {str(e)}")

@commit_router.get("/github/{owner}/{repo}/commits/{sha}/files")
async def get_commit_files_metadata(
    owner: str,
    repo: str,
    sha: str,
    request: Request = None
):
    """
    Get detailed file metadata for a specific commit from GitHub API
    """
    try:
        from services.github_service import fetch_commit_files_metadata
        
        # Get user token if available
        token = None
        if hasattr(request, 'headers'):
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith(("Bearer ", "token ")):
                token = auth_header.split(" ", 1)[1]
        
        # Fetch file metadata
        file_metadata = await fetch_commit_files_metadata(owner, repo, sha, token)
        
        return {
            "success": True,
            "repository": f"{owner}/{repo}",
            "commit_sha": sha,
            "file_metadata": file_metadata
        }
        
    except Exception as e:
        logger.error(f"Error fetching commit files metadata for {sha}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch file metadata: {str(e)}")

@commit_router.post("/github/{owner}/{repo}/sync-enhanced")
async def sync_enhanced_commits_from_github(
    owner: str,
    repo: str,
    branch: str = Query("main", description="Branch name"),
    max_commits: int = Query(100, ge=1, le=500, description="Maximum commits to sync"),
    force_update: bool = Query(True, description="Force update existing commits"),
    request: Request = None
):
    """
    Sync commits from GitHub with enhanced metadata to database
    """
    try:
        from services.github_service import fetch_enhanced_commits_batch
        from services.commit_service import save_multiple_commits, get_repo_id_by_owner_and_name
        from services.branch_service import get_or_create_branch
        
        # Get repository ID
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo} not found")
        
        # Get user token
        token = None
        if hasattr(request, 'headers'):
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith(("Bearer ", "token ")):
                token = auth_header.split(" ", 1)[1]
        
        # Fetch enhanced commits from GitHub
        enhanced_commits = await fetch_enhanced_commits_batch(
            owner=owner,
            repo=repo,
            branch=branch,
            token=token,
            max_commits=max_commits
        )
        
        if not enhanced_commits:
            return {
                "success": True,
                "message": "No commits found to sync",
                "synced_count": 0
            }
        
        # Get or create branch
        branch_info = await get_or_create_branch(repo_id, branch)
        branch_id = branch_info.get("id") if branch_info else None
        
        # Prepare commits for saving with enhanced metadata
        commits_to_save = []
        for commit in enhanced_commits:
            commit_data = {
                **commit,
                "repo_id": repo_id,
                "branch_id": branch_id,
                "branch_name": branch
            }
            commits_to_save.append(commit_data)
        
        # Save commits to database with force_update
        processed_count = await save_multiple_commits(
            commits_data=commits_to_save, 
            repo_id=repo_id, 
            branch_name=branch, 
            branch_id=branch_id,
            force_update=force_update
        )
        
        return {
            "success": True,
            "repository": f"{owner}/{repo}",
            "branch": branch,
            "commits_processed": processed_count,
            "total_fetched": len(enhanced_commits),
            "force_update": force_update,
            "enhanced_metadata": {
                "files_changed": True,
                "file_analysis": True,
                "directory_tracking": True,
                "merge_detection": True,
                "change_statistics": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error syncing enhanced commits for {owner}/{repo}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to sync enhanced commits: {str(e)}")

@commit_router.get("/github/{owner}/{repo}/compare/{base}...{head}")
async def compare_commits_github(
    owner: str,
    repo: str,
    base: str,
    head: str,
    token: str = Depends(security)
):
    """
    Compare two commits using GitHub API to get detailed change information
    """
    try:
        from services.github_service import get_commit_comparison
        comparison_data = await get_commit_comparison(owner, repo, base, head, token)
        return comparison_data
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error comparing commits {base}...{head} for {owner}/{repo}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compare commits: {str(e)}")



