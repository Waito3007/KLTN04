# backend/api/routes/sync.py
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
import httpx
import asyncio
import logging
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from services.repo_service import save_repository, get_repo_id_by_owner_and_name
from services.branch_service import sync_branches_for_repo
from services.commit_service import save_commit, save_commit_with_diff
from services.issue_service import save_issue
from services.pull_request_service import save_pull_request
from services.github_service import fetch_commit_details, fetch_branch_stats
import time

# Import sync events để có thể emit events real-time
from .sync_events import emit_sync_start, emit_sync_progress, emit_sync_complete, emit_sync_error

sync_router = APIRouter()
logger = logging.getLogger(__name__)

# Constants
GITHUB_API_BASE = "https://api.github.com"
MAX_CONCURRENT_REQUESTS = 10  # Limit concurrent requests to avoid rate limiting
BATCH_SIZE = 50  # Process commits in batches

async def github_api_call_batch(urls: List[str], token: str, semaphore: asyncio.Semaphore) -> List[Dict[str, Any]]:
    """
    Gọi nhiều GitHub API URLs đồng thời với semaphore để limit concurrent requests
    
    Args:
        urls: List các GitHub API URLs
        token: Authorization token
        semaphore: Semaphore để limit concurrent requests
    
    Returns:
        List response JSON data
    """
    headers = {
        "Authorization": token,
        "Accept": "application/vnd.github+json", 
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    async def fetch_single(url: str, client: httpx.AsyncClient) -> Dict[str, Any]:
        async with semaphore:
            try:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    # Rate limit - wait a bit and return None
                    await asyncio.sleep(1)
                    return None
                else:
                    logger.warning(f"API call failed for {url}: {resp.status_code}")
                    return None
            except Exception as e:
                logger.warning(f"Exception in API call for {url}: {e}")
                return None
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [fetch_single(url, client) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        valid_results = []
        for result in results:
            if result is not None and not isinstance(result, Exception):
                valid_results.append(result)
        
        return valid_results

async def github_api_call(url: str, token: str, retries: int = 3) -> Dict[str, Any]:
    """
    Gọi GitHub API với error handling và retry logic
    
    Args:
        url: GitHub API URL
        token: Authorization token
        retries: Số lần retry nếu rate limit
    
    Returns:
        Response JSON data
    
    Raises:
        HTTPException: Khi API call thất bại
    """
    headers = {
        "Authorization": token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    for attempt in range(retries + 1):
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(url, headers=headers)
                
                # Handle rate limiting
                if resp.status_code == 429:
                    if attempt < retries:
                        reset_time = int(resp.headers.get("X-RateLimit-Reset", "0"))
                        wait_time = min(reset_time - int(asyncio.get_event_loop().time()), 60)
                        logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}")
                        await asyncio.sleep(max(wait_time, 1))
                        continue
                    else:
                        raise HTTPException(
                            status_code=429, 
                            detail="GitHub API rate limit exceeded. Please try again later."
                        )
                
                # Handle other HTTP errors
                if resp.status_code != 200:
                    error_detail = f"GitHub API error: {resp.status_code}"
                    try:
                        error_data = resp.json()
                        error_detail += f" - {error_data.get('message', resp.text)}"
                    except:
                        error_detail += f" - {resp.text}"
                    
                    raise HTTPException(status_code=resp.status_code, detail=error_detail)
                
                return resp.json()
                
            except httpx.TimeoutException:
                if attempt < retries:
                    logger.warning(f"Request timeout, retrying... (attempt {attempt + 1})")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise HTTPException(status_code=408, detail="GitHub API request timeout")
            
            except httpx.RequestError as e:
                if attempt < retries:
                    logger.warning(f"Request error: {e}, retrying... (attempt {attempt + 1})")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise HTTPException(status_code=500, detail=f"GitHub API request failed: {str(e)}")
    
    raise HTTPException(status_code=500, detail="All retry attempts failed")

@sync_router.post("/github/{owner}/{repo}/sync-all")
async def sync_all_optimized(owner: str, repo: str, request: Request, background_tasks: BackgroundTasks):
    """
    Đồng bộ toàn bộ dữ liệu repository với tối ưu tốc độ:
    - Repository information
    - All branches (concurrent)
    - All commits from all branches (batch processing) 
    - All issues (batch processing)
    - All pull requests (batch processing)
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    # Start background sync task
    background_tasks.add_task(
        sync_all_background_optimized,
        owner,
        repo, 
        token
    )
    
    return {
        "status": "accepted",
        "message": f"Started optimized sync for {owner}/{repo}",
        "repository": f"{owner}/{repo}",
        "note": "Sync is running in background. Check logs for progress."
    }

async def sync_all_background_optimized(owner: str, repo: str, token: str):
    """
    Background task để đồng bộ toàn bộ repository với tối ưu tốc độ
    """
    start_time = time.time()
    repo_key = f"{owner}/{repo}"
    logger.info(f"🚀 Starting optimized sync for {repo_key}")
    
    # Emit sync start event
    await emit_sync_start(repo_key, "optimized")
    
    sync_results = {
        "repository_synced": False,
        "branches_synced": 0,
        "commits_synced": 0,
        "issues_synced": 0,
        "pull_requests_synced": 0,
        "errors": [],
        "timing": {}
    }
    
    try:
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        # 1. Sync repository (fastest)
        await emit_sync_progress(repo_key, 1, 5, "Syncing repository metadata")
        step_start = time.time()
        
        # 1. Sync repository (fastest)
        step_start = time.time()
        repo_data = await github_api_call(f"https://api.github.com/repos/{owner}/{repo}", token)
        repo_entry = {
            "github_id": repo_data["id"],
            "name": repo_data["name"],
            "owner": repo_data["owner"]["login"],
            "description": repo_data["description"],
            "stars": repo_data["stargazers_count"],
            "forks": repo_data["forks_count"],
            "language": repo_data["language"],
            "open_issues": repo_data["open_issues_count"],
            "url": repo_data["html_url"],
            "full_name": repo_data.get("full_name"),
            "clone_url": repo_data.get("clone_url"),
            "is_private": repo_data.get("private", False),
            "is_fork": repo_data.get("fork", False),
            "default_branch": repo_data.get("default_branch", "main"),
            "sync_status": "sync_optimized",
        }
        await save_repository(repo_entry)
        sync_results["repository_synced"] = True
        sync_results["timing"]["repository"] = time.time() - step_start
        logger.info(f"✅ Repository synced in {sync_results['timing']['repository']:.2f}s")
        
        # 2. Get repo_id
        await emit_sync_progress(repo_key, 2, 5, "Getting repository ID")
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise Exception("Repository not found after creation")
        
        # 3. Sync branches concurrently
        await emit_sync_progress(repo_key, 3, 5, "Syncing branches")
        step_start = time.time()
        branches_data = await github_api_call(f"https://api.github.com/repos/{owner}/{repo}/branches", token)
        default_branch = repo_data.get("default_branch", "main")
        
        # Prepare branch data
        branches_to_save = []
        for branch in branches_data:
            branch_info = {
                "name": branch["name"],
                "sha": branch.get("commit", {}).get("sha"),
                "is_default": branch["name"] == default_branch,
                "is_protected": branch.get("protected", False),
                "repo_id": repo_id
            }
            branches_to_save.append(branch_info)
        
        branches_synced = await sync_branches_for_repo(
            repo_id, 
            branches_to_save, 
            default_branch=default_branch,
            replace_existing=False  # Don't replace to avoid foreign key issues
        )
        sync_results["branches_synced"] = branches_synced
        sync_results["timing"]["branches"] = time.time() - step_start
        logger.info(f"✅ {branches_synced} branches synced in {sync_results['timing']['branches']:.2f}s")
        
        # 4. Sync commits with batch processing and concurrent diff fetching
        await emit_sync_progress(repo_key, 4, 5, "Syncing commits")
        step_start = time.time()
        commits_synced = await sync_commits_batch_optimized(
            owner, repo, repo_id, branches_data, token, semaphore
        )
        sync_results["commits_synced"] = commits_synced
        sync_results["timing"]["commits"] = time.time() - step_start
        logger.info(f"✅ {commits_synced} commits synced in {sync_results['timing']['commits']:.2f}s")
        
        # 5. Sync issues and PRs concurrently
        await emit_sync_progress(repo_key, 5, 5, "Syncing issues and pull requests")
        step_start = time.time()
        issues_task = asyncio.create_task(
            sync_issues_batch_optimized(owner, repo, repo_id, token, semaphore)
        )
        prs_task = asyncio.create_task(
            sync_prs_batch_optimized(owner, repo, repo_id, token, semaphore)
        )
        
        # Wait for both to complete
        issues_synced, prs_synced = await asyncio.gather(issues_task, prs_task)
        
        sync_results["issues_synced"] = issues_synced
        sync_results["pull_requests_synced"] = prs_synced
        sync_results["timing"]["issues_and_prs"] = time.time() - step_start
        logger.info(f"✅ {issues_synced} issues and {prs_synced} PRs synced in {sync_results['timing']['issues_and_prs']:.2f}s")
        
        try:
            # Update final status
            logger.debug(f"🔧 About to save repository with final status")
            await save_repository({
                **repo_entry,
                "sync_status": "sync_completed"
            })
            logger.debug(f"✅ Repository saved with final status")
            
            # Update repository sync status and last_synced timestamp
            logger.debug(f"🔧 About to update repo sync status")
            from services.repo_service import update_repo_sync_status
            await update_repo_sync_status(owner, repo, "completed")
            logger.debug(f"✅ Repo sync status updated")
            
            total_time = time.time() - start_time
            sync_results["timing"]["total"] = total_time
            
            # Emit sync complete event
            logger.debug(f"🔧 About to emit sync complete event")
            await emit_sync_complete(repo_key, True, sync_results)
            logger.debug(f"✅ Sync complete event emitted")
            
            logger.info(f"🎉 Optimized sync completed for {repo_key} in {total_time:.2f}s")
            logger.info(f"📊 Results: {sync_results}")
            
        except Exception as final_error:
            logger.error(f"❌ Error in final sync steps for {repo_key}: {str(final_error)}")
            logger.error(f"❌ Error type: {type(final_error).__name__}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise final_error
        
    except Exception as e:
        logger.error(f"❌ Error in optimized sync for {repo_key}: {str(e)}")
        sync_results["errors"].append(f"Fatal error: {str(e)}")
        
        # Emit sync error event
        await emit_sync_error(repo_key, str(e), "sync_all_background")

async def sync_commits_batch_optimized(
    owner: str, 
    repo: str, 
    repo_id: int, 
    branches_data: List[Dict], 
    token: str,
    semaphore: asyncio.Semaphore
) -> int:
    """
    Đồng bộ commits với batch processing và concurrent diff fetching
    """
    total_commits_synced = 0
    github_token = token.replace("token ", "") if token.startswith("token ") else None
    
    for branch in branches_data:
        branch_name = branch["name"]
        logger.info(f"🔄 Processing commits from branch: {branch_name}")
        
        # Get all commits for this branch first (lightweight)
        all_commits = []
        page = 1
        per_page = 100
        
        while True:
            commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits?sha={branch_name}&per_page={per_page}&page={page}"
            commits_data = await github_api_call(commits_url, token)
            
            if not commits_data:
                break
                
            all_commits.extend(commits_data)
            page += 1
            
            if len(commits_data) < per_page:
                break
        
        logger.info(f"📝 Found {len(all_commits)} commits in branch {branch_name}")
        
        # Process commits in batches
        for i in range(0, len(all_commits), BATCH_SIZE):
            batch = all_commits[i:i + BATCH_SIZE]
            batch_start = time.time()
            
            # Process batch concurrently
            tasks = []
            for commit in batch:
                task = asyncio.create_task(
                    process_single_commit_optimized(
                        commit, owner, repo, repo_id, branch_name, github_token, semaphore
                    )
                )
                tasks.append(task)
            
            # Wait for batch to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful commits
            batch_success = sum(1 for r in results if r is True)
            total_commits_synced += batch_success
            
            batch_time = time.time() - batch_start
            logger.info(f"✅ Batch {i//BATCH_SIZE + 1}: {batch_success}/{len(batch)} commits in {batch_time:.2f}s")
    
    return total_commits_synced

async def process_single_commit_optimized(
    commit: Dict, 
    owner: str, 
    repo: str, 
    repo_id: int, 
    branch_name: str, 
    github_token: str,
    semaphore: asyncio.Semaphore
) -> bool:
    """
    Process single commit with optimized diff fetching
    """
    try:
        commit_data = {
            "sha": commit["sha"],
            "repo_id": repo_id,
            "branch_name": branch_name,
            "commit": commit.get("commit", {}),
            "parents": commit.get("parents", [])
        }
        
        # Use optimized save function
        commit_id = await save_commit_with_diff(
            commit_data, 
            owner, 
            repo, 
            github_token, 
            force_update=False
        )
        
        return commit_id is not None
        
    except Exception as e:
        logger.warning(f"⚠️ Error processing commit {commit.get('sha', 'unknown')}: {e}")
        return False

async def sync_issues_batch_optimized(
    owner: str, 
    repo: str, 
    repo_id: int, 
    token: str,
    semaphore: asyncio.Semaphore
) -> int:
    """
    Đồng bộ issues với batch processing
    """
    total_issues = 0
    page = 1
    per_page = 100
    
    while True:
        issues_url = f"https://api.github.com/repos/{owner}/{repo}/issues?state=all&per_page={per_page}&page={page}"
        issues_data = await github_api_call(issues_url, token)
        
        if not issues_data:
            break
        
        # Filter out pull requests
        actual_issues = [issue for issue in issues_data if "pull_request" not in issue]
        
        # Process issues concurrently
        tasks = []
        for issue in actual_issues:
            task = asyncio.create_task(
                process_single_issue_optimized(issue, repo_id, semaphore)
            )
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            total_issues += success_count
        
        page += 1
        if len(issues_data) < per_page:
            break
    
    return total_issues

async def process_single_issue_optimized(issue: Dict, repo_id: int, semaphore: asyncio.Semaphore) -> bool:
    """
    Process single issue
    """
    async with semaphore:
        try:
            issue_entry = {
                "github_id": issue["id"],
                "repo_id": repo_id,
                "number": issue["number"],
                "title": issue["title"],
                "body": issue.get("body", ""),
                "state": issue["state"],
                "author": issue["user"]["login"],
                "created_at": issue["created_at"],
                "updated_at": issue["updated_at"],
                "url": issue["html_url"]
            }
            await save_issue(issue_entry)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Error processing issue {issue.get('id', 'unknown')}: {e}")
            return False

async def sync_prs_batch_optimized(
    owner: str, 
    repo: str, 
    repo_id: int, 
    token: str,
    semaphore: asyncio.Semaphore
) -> int:
    """
    Đồng bộ pull requests với batch processing
    """
    total_prs = 0
    page = 1
    per_page = 100
    
    while True:
        prs_url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=all&per_page={per_page}&page={page}"
        prs_data = await github_api_call(prs_url, token)
        
        if not prs_data:
            break
        
        # Process PRs concurrently
        tasks = []
        for pr in prs_data:
            task = asyncio.create_task(
                process_single_pr_optimized(pr, repo_id, semaphore)
            )
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            total_prs += success_count
        
        page += 1
        if len(prs_data) < per_page:
            break
    
    return total_prs

async def process_single_pr_optimized(pr: Dict, repo_id: int, semaphore: asyncio.Semaphore) -> bool:
    """
    Process single pull request
    """
    async with semaphore:
        try:
            pr_entry = {
                "github_id": pr["id"],
                "repo_id": repo_id,
                "number": pr["number"],
                "title": pr["title"],
                "body": pr.get("body", ""),
                "state": pr["state"],
                "author": pr["user"]["login"],
                "created_at": pr["created_at"],
                "updated_at": pr["updated_at"],
                "merged_at": pr.get("merged_at"),
                "url": pr["html_url"],
                "base_branch": pr["base"]["ref"],
                "head_branch": pr["head"]["ref"]
            }
            await save_pull_request(pr_entry)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Error processing PR {pr.get('id', 'unknown')}: {e}")
            return False

# Endpoint đồng bộ nhanh - chỉ thông tin cơ bản
@sync_router.post("/github/{owner}/{repo}/sync-basic")
async def sync_basic(owner: str, repo: str, request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    try:
        # Chỉ đồng bộ repository
        repo_data = await github_api_call(f"https://api.github.com/repos/{owner}/{repo}", token)
        repo_entry = {
            "github_id": repo_data["id"],
            "name": repo_data["name"],
            "owner": repo_data["owner"]["login"],
            "description": repo_data["description"],
            "stars": repo_data["stargazers_count"],
            "forks": repo_data["forks_count"],
            "language": repo_data["language"],
            "open_issues": repo_data["open_issues_count"],
            "url": repo_data["html_url"],
            "full_name": repo_data.get("full_name"),
            "clone_url": repo_data.get("clone_url"),
            "is_private": repo_data.get("private", False),
            "is_fork": repo_data.get("fork", False),
            "default_branch": repo_data.get("default_branch", "main"),
            "sync_status": "completed",
        }
        await save_repository(repo_entry)
        
        # Update repository sync status and last_synced timestamp
        from services.repo_service import update_repo_sync_status
        await update_repo_sync_status(owner, repo, "completed")
        
        return {"message": f"Đồng bộ cơ bản {owner}/{repo} thành công!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đồng bộ cơ bản {owner}/{repo}: {str(e)}")

# Endpoint đồng bộ nâng cao - bao gồm thông tin commit chi tiết
@sync_router.post("/github/{owner}/{repo}/sync-enhanced")
async def sync_enhanced(owner: str, repo: str, request: Request):
    """
    Đồng bộ repository với thông tin chi tiết bao gồm:
    - Thông tin repository đầy đủ
    - Thông tin branch đầy đủ
    - Thông tin commit cuối cùng cho mỗi branch
    - Thống kê branch (nếu có)
    
    Lưu ý: Endpoint này sẽ chậm hơn do phải gọi nhiều API calls
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    try:
        logger.info(f"Starting enhanced sync for {owner}/{repo}")
        
        # 1. Sync repository
        repo_data = await github_api_call(f"https://api.github.com/repos/{owner}/{repo}", token)
        repo_entry = {
            "github_id": repo_data["id"],
            "name": repo_data["name"],
            "owner": repo_data["owner"]["login"],
            "description": repo_data["description"],
            "stars": repo_data["stargazers_count"],
            "forks": repo_data["forks_count"],
            "language": repo_data["language"],
            "open_issues": repo_data["open_issues_count"],
            "url": repo_data["html_url"],
            "full_name": repo_data.get("full_name"),
            "clone_url": repo_data.get("clone_url"),
            "is_private": repo_data.get("private", False),
            "is_fork": repo_data.get("fork", False),
            "default_branch": repo_data.get("default_branch", "main"),
            "sync_status": "enhanced_completed",
        }
        await save_repository(repo_entry)
        
        # 2. Sync branches với thông tin chi tiết
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        branches_data = await github_api_call(f"https://api.github.com/repos/{owner}/{repo}/branches", token)
        
        default_branch = repo_data.get("default_branch", "main")
        branches_to_save = []
        
        # Process branches with enhanced data
        for i, branch in enumerate(branches_data):
            logger.info(f"Processing branch {i+1}/{len(branches_data)}: {branch['name']}")
            
            branch_info = {
                "name": branch["name"],
                "sha": branch.get("commit", {}).get("sha"),
                "is_default": branch["name"] == default_branch,
                "is_protected": branch.get("protected", False),
            }
            
            # Lấy thông tin commit chi tiết cho branch
            if branch_info["sha"]:
                try:
                    commit_details = await fetch_commit_details(
                        branch_info["sha"], owner, repo, token
                    )
                    if commit_details:
                        branch_info.update({
                            "last_commit_date": commit_details.get("date"),
                            "last_committer_name": commit_details.get("committer_name"),
                            "creator_name": commit_details.get("author_name"),  # Assuming first commit author as creator
                        })
                except Exception as e:
                    logger.warning(f"Failed to fetch commit details for branch {branch['name']}: {e}")
            
            # Lấy thống kê branch (optional)
            try:
                branch_stats = await fetch_branch_stats(owner, repo, branch["name"], token)
                if branch_stats:
                    branch_info.update({
                        "commits_count": branch_stats.get("commits_count"),
                        "contributors_count": branch_stats.get("contributors_count"),
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch branch stats for {branch['name']}: {e}")
            
            branches_to_save.append(branch_info)
            
            # Add small delay to avoid hitting rate limits too hard
            if i < len(branches_data) - 1:  # Don't sleep after last branch
                await asyncio.sleep(0.1)
        
        # Đồng bộ hóa hàng loạt với dữ liệu đầy đủ
        branches_synced = await sync_branches_for_repo(
            repo_id, 
            branches_to_save, 
            default_branch=default_branch,
            replace_existing=True
        )
        
        logger.info(f"Enhanced sync completed for {owner}/{repo}: {branches_synced} branches synced")
        
        # Update repository sync status and last_synced timestamp
        from services.repo_service import update_repo_sync_status
        await update_repo_sync_status(owner, repo, "completed")
        
        return {
            "message": f"Đồng bộ nâng cao {owner}/{repo} thành công!",
            "repository_synced": True,
            "branches_synced": branches_synced,
            "enhanced_data": True
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Enhanced sync error for {owner}/{repo}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi đồng bộ nâng cao {owner}/{repo}: {str(e)}")

# Endpoint kiểm tra trạng thái GitHub API và token
@sync_router.get("/github/status")
async def github_status(request: Request):
    """
    Kiểm tra trạng thái kết nối GitHub API và thông tin rate limit
    
    Returns:
        dict: Thông tin về token, rate limit, và trạng thái API
    """
    from services.github_service import validate_github_token, get_rate_limit_info
    
    token = request.headers.get("Authorization", "").replace("token ", "")
    
    result = {
        "github_api_accessible": False,
        "token_valid": False,
        "rate_limit": None,
        "token_provided": bool(token)
    }
    
    try:
        # Kiểm tra token nếu được cung cấp
        if token:
            result["token_valid"] = await validate_github_token(token)
        
        # Lấy thông tin rate limit
        rate_limit_info = await get_rate_limit_info(token if token else None)
        result["rate_limit"] = rate_limit_info.get("resources", {}).get("core", {})
        result["github_api_accessible"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

# Endpoint lấy danh sách repositories có sẵn cho user
@sync_router.get("/github/repositories")
async def list_user_repositories(request: Request, per_page: int = 30, page: int = 1):
    """
    Lấy danh sách repositories của user hiện tại
    
    Args:
        per_page: Số repo trên mỗi trang (max 100)
        page: Số trang
    
    Returns:
        list: Danh sách repositories
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    try:
        # Giới hạn per_page để tránh quá tải
        per_page = min(max(per_page, 1), 100)
        page = max(page, 1)
        
        url = f"https://api.github.com/user/repos?per_page={per_page}&page={page}&sort=updated"
        repos_data = await github_api_call(url, token)
        
        # Trả về thông tin cơ bản của các repos
        simplified_repos = []
        for repo in repos_data:
            simplified_repos.append({
                "id": repo["id"],
                "name": repo["name"],
                "full_name": repo["full_name"],
                "owner": repo["owner"]["login"],
                "description": repo.get("description"),
                "language": repo.get("language"),
                "stars": repo["stargazers_count"],
                "forks": repo["forks_count"],
                "updated_at": repo["updated_at"],
                "is_private": repo["private"],
                "is_fork": repo["fork"],
                "default_branch": repo.get("default_branch", "main")
            })
        
        return {
            "repositories": simplified_repos,
            "page": page,
            "per_page": per_page,
            "total_returned": len(simplified_repos)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lấy danh sách repositories: {str(e)}")

# Endpoint thống kê repository và branches
@sync_router.get("/github/{owner}/{repo}/stats")
async def get_repository_stats(owner: str, repo: str):
    """
    Lấy thống kê chi tiết về repository và branches
    
    Returns:
        dict: Thống kê repository và branches
    """
    from services.branch_service import get_branch_statistics, find_stale_branches, get_most_active_branches
    from services.repo_service import get_repository_by_owner_and_name
    
    try:
        # Lấy thông tin repository
        repo_info = await get_repository_by_owner_and_name(owner, repo)
        if not repo_info:
            raise HTTPException(status_code=404, detail="Repository not found in database")
        
        repo_id = repo_info['id']
        
        # Lấy thống kê branches
        branch_stats = await get_branch_statistics(repo_id)
        
        # Lấy branches cũ (90 ngày)
        stale_branches = await find_stale_branches(repo_id, days_threshold=90)
        
        # Lấy branches hoạt động nhất
        active_branches = await get_most_active_branches(repo_id, limit=5)
        
        return {
            "repository": {
                "name": repo_info['name'],
                "owner": repo_info['owner'],
                "stars": repo_info.get('stars', 0),
                "forks": repo_info.get('forks', 0),
                "language": repo_info.get('language'),
                "last_synced": repo_info.get('updated_at'),
                "sync_status": repo_info.get('sync_status', 'unknown')
            },
            "branch_statistics": branch_stats,
            "stale_branches": {
                "count": len(stale_branches),
                "branches": [{"name": b["name"], "last_commit_date": b["last_commit_date"]} for b in stale_branches[:10]]
            },
            "most_active_branches": [
                {
                    "name": b["name"], 
                    "commits_count": b["commits_count"],
                    "is_default": b["is_default"],
                    "is_protected": b["is_protected"]
                } 
                for b in active_branches
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lấy thống kê {owner}/{repo}: {str(e)}")

# Endpoint để cập nhật branch metadata
@sync_router.patch("/github/{owner}/{repo}/branches/{branch_name}")
async def update_branch_info(owner: str, repo: str, branch_name: str, metadata: dict, request: Request):
    """
    Cập nhật thông tin metadata của một branch
    
    Args:
        owner: Chủ sở hữu repository
        repo: Tên repository
        branch_name: Tên branch
        metadata: Dữ liệu cần cập nhật
    
    Returns:
        dict: Kết quả cập nhật
    """
    from services.branch_service import update_branch_metadata
    from services.repo_service import get_repo_id_by_owner_and_name
    
    # Optional: Kiểm tra token nếu cần authorization
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    try:
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        success = await update_branch_metadata(repo_id, branch_name, metadata)
        
        if success:
            return {
                "message": f"Branch {branch_name} updated successfully",
                "repository": f"{owner}/{repo}",
                "branch": branch_name,
                "updated_fields": list(metadata.keys())
            }
        else:
            raise HTTPException(status_code=404, detail="Branch not found or no valid fields to update")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi cập nhật branch {branch_name}: {str(e)}")

# ==================== AUTO-SYNC ENDPOINTS FOR REPOSITORY SELECTION ====================

@sync_router.post("/github/{owner}/{repo}/sync-branches")
async def sync_repository_branches(owner: str, repo: str, request: Request):
    """
    Sync branches for specific repository when user selects it
    Auto-creates repository if not exists
    """
    token = request.headers.get("Authorization")
    if not token or not token.startswith("token "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    try:
        # Get repository ID, create if not exists
        repo_id = await get_repo_id_by_owner_and_name(owner, repo)
        if not repo_id:
            # Auto-create repository
            repo_data = await github_api_call(f"{GITHUB_API_BASE}/repos/{owner}/{repo}", token)
            repo_entry = {
                "github_id": repo_data["id"],
                "name": repo_data["name"],
                "owner": repo_data["owner"]["login"],
                "description": repo_data.get("description"),
                "full_name": repo_data.get("full_name"),
                "default_branch": repo_data.get("default_branch", "main"),
                "stars": repo_data.get("stargazers_count", 0),
                "forks": repo_data.get("forks_count", 0),
                "language": repo_data.get("language"),
                "is_private": repo_data.get("private", False),
                "sync_status": "auto_created"
            }
            await save_repository(repo_entry)
            repo_id = await get_repo_id_by_owner_and_name(owner, repo)

        # Sync branches
        branches_data = await github_api_call(f"{GITHUB_API_BASE}/repos/{owner}/{repo}/branches", token)
        
        # Enhanced branch data with commit info
        enhanced_branches = []
        default_branch = None
        
        # Get repository info for default branch
        try:
            repo_info = await github_api_call(f"{GITHUB_API_BASE}/repos/{owner}/{repo}", token)
            default_branch = repo_info.get("default_branch", "main")
        except:
            default_branch = "main"
        
        for branch in branches_data:
            try:
                # Get additional commit info for each branch
                commit_data = await github_api_call(
                    f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{branch['commit']['sha']}", 
                    token
                )
                
                enhanced_branch = {
                    "name": branch["name"],
                    "sha": branch["commit"]["sha"],
                    "is_protected": branch.get("protected", False),
                    "is_default": branch["name"] == default_branch,
                    "repo_id": repo_id,
                    "creator_user_id": None,  # Could enhance later
                    "last_committer_user_id": None,  # Could enhance later
                    "commits_count": 1,  # Basic count, could enhance
                    "contributors_count": 1,  # Basic count, could enhance
                    "last_commit_date": commit_data["commit"]["committer"]["date"]
                }
                enhanced_branches.append(enhanced_branch)
                
            except Exception as e:
                logger.warning(f"Error getting commit info for branch {branch['name']}: {e}")
                # Fallback to basic branch info
                enhanced_branches.append({
                    "name": branch["name"],
                    "sha": branch["commit"]["sha"],
                    "is_protected": branch.get("protected", False),
                    "is_default": branch["name"] == default_branch,
                    "repo_id": repo_id,
                    "commits_count": 1,
                    "contributors_count": 1
                })        # Save branches to database
        saved_count = await save_multiple_branches(repo_id, enhanced_branches)
        
        return {
            "status": "success",
            "repository": f"{owner}/{repo}",
            "branches": enhanced_branches,
            "saved_count": saved_count,
            "message": f"Successfully synced {saved_count} branches"
        }
        
    except Exception as e:
        logger.error(f"Error syncing branches for {owner}/{repo}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to sync branches: {str(e)}")

# ==================== HELPER FUNCTIONS ====================

async def save_multiple_branches(repo_id: int, branches_list: list):
    """Save multiple branches efficiently"""
    if not branches_list:
        return 0
    
    try:
        from db.models.branches import branches
        from db.database import engine
        from sqlalchemy import insert, delete
        
        with engine.connect() as conn:
            # Clear existing branches for this repository
            delete_query = delete(branches).where(branches.c.repo_id == repo_id)
            conn.execute(delete_query)
            
            # Insert new branches
            if branches_list:
                insert_query = insert(branches)
                conn.execute(insert_query, branches_list)
            
            conn.commit()
        
        return len(branches_list)
        
    except Exception as e:
        logger.error(f"Error saving branches: {e}")
        return 0
