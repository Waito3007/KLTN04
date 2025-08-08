# backend/services/github_service.py
# Service xử lý các tương tác với GitHub API

import httpx
import os
import re
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_URL = "https://api.github.com"

# Default headers for GitHub API requests
def get_headers(token: str = None) -> Dict[str, str]:
    """Tạo headers chuẩn cho GitHub API request"""
    auth_token = token or GITHUB_TOKEN
    return {
        "Authorization": f"Bearer {auth_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

async def fetch_from_github(url: str, token: str = None) -> Dict[str, Any]:
    """
    Hàm tổng quát để fetch dữ liệu từ GitHub API
    
    Args:
        url (str): Phần cuối của URL (sau BASE_URL)
        token (str): GitHub token (optional)
    
    Returns:
        dict: Dữ liệu JSON trả về từ GitHub API
    
    Raises:
        HTTPError: Nếu request lỗi
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}{url}", headers=get_headers(token))
        response.raise_for_status()
        return response.json()

async def fetch_commits(
    token: str, 
    owner: str, 
    name: str, 
    branch: str, 
    since: Optional[str] = None, 
    until: Optional[str] = None,
    per_page: int = 100,
    include_details: bool = False
) -> List[Dict[str, Any]]:
    """
    Lấy danh sách commit từ repository GitHub với enhanced metadata
    
    Args:
        token (str): GitHub access token
        owner (str): Chủ repository
        name (str): Tên repository
        branch (str): Tên branch
        since (Optional[str]): Lọc commit từ thời gian này (ISO format)
        until (Optional[str]): Lọc commit đến thời gian này (ISO format)
        per_page (int): Số commits per page (max 100)
        include_details (bool): Có lấy chi tiết files và stats không
    
    Returns:
        list: Danh sách commit với enhanced metadata
    
    Raises:
        HTTPError: Nếu request lỗi
    """
    url = f"/repos/{owner}/{name}/commits"
    
    # Parameters cho request
    params = {"sha": branch, "per_page": min(per_page, 100)}
    
    # Thêm tham số lọc thời gian nếu có
    if since:
        params["since"] = since
    if until:
        params["until"] = until

    # Gọi API GitHub
    async with httpx.AsyncClient() as client:
        full_url = f"{BASE_URL}{url}"
        response = await client.get(full_url, headers=get_headers(token), params=params)
        response.raise_for_status()
        commits_data = response.json()
        
        if not include_details:
            return commits_data
        
        # Fetch detailed information for each commit
        enhanced_commits = []
        for commit in commits_data:
            commit_sha = commit.get("sha")
            if commit_sha:
                detailed_commit = await fetch_commit_details(commit_sha, owner, name, token)
                if detailed_commit:
                    enhanced_commits.append(detailed_commit)
                else:
                    # Fallback to basic commit data with extracted metadata
                    enhanced_commit = extract_basic_commit_metadata(commit)
                    enhanced_commits.append(enhanced_commit)
            
        return enhanced_commits

async def fetch_commit_details(commit_sha: str, owner: str, repo: str, token: str = None) -> Optional[Dict[str, Any]]:
    """
    Lấy thông tin chi tiết của một commit với enhanced metadata
    
    Args:
        commit_sha (str): SHA hash của commit
        owner (str): Chủ sở hữu repo
        repo (str): Tên repository
        token (str): GitHub token (optional)
    
    Returns:
        dict: Thông tin chi tiết commit với enhanced metadata hoặc None nếu lỗi
    """
    try:
        url = f"/repos/{owner}/{repo}/commits/{commit_sha}"
        
        async with httpx.AsyncClient() as client:
            full_url = f"{BASE_URL}{url}"
            response = await client.get(full_url, headers=get_headers(token))
            
            if response.status_code == 200:
                commit_data = response.json()
                
                # Extract basic commit info
                commit_info = commit_data.get("commit", {})
                author_info = commit_info.get("author", {})
                committer_info = commit_info.get("committer", {})
                stats = commit_data.get("stats", {})
                
                # Extract files information
                files = commit_data.get("files", [])
                modified_files = []
                file_types = {}
                modified_directories = {}
                
                for file_info in files:
                    filename = file_info.get("filename", "")
                    if filename:
                        modified_files.append(filename)
                        
                        # Extract file extension
                        if "." in filename:
                            ext = "." + filename.split(".")[-1].lower()
                            file_types[ext] = file_types.get(ext, 0) + 1
                        
                        # Extract directory
                        if "/" in filename:
                            directory = "/".join(filename.split("/")[:-1])
                            modified_directories[directory] = modified_directories.get(directory, 0) + 1
                        else:
                            modified_directories["root"] = modified_directories.get("root", 0) + 1
                
                # Check if this is a merge commit
                parents = commit_data.get("parents", [])
                is_merge = len(parents) > 1
                
                # Calculate enhanced metadata
                additions = stats.get("additions", 0)
                deletions = stats.get("deletions", 0)
                total_changes = additions + deletions
                files_changed = len(files)
                
                return {
                    "sha": commit_data.get("sha"),
                    "date": author_info.get("date"),
                    "committer_date": committer_info.get("date"),
                    "author_name": author_info.get("name"),
                    "author_email": author_info.get("email"),
                    "committer_name": committer_info.get("name"),
                    "committer_email": committer_info.get("email"),
                    "message": commit_info.get("message"),
                    "url": commit_data.get("html_url"),
                    "parents": parents,
                    # Enhanced metadata
                    "stats": {
                        "additions": additions,
                        "deletions": deletions,
                        "total": stats.get("total", 0)
                    },
                    "files_changed": files_changed,
                    "additions": additions,
                    "deletions": deletions,
                    "total_changes": total_changes,
                    "is_merge": is_merge,
                    "modified_files": modified_files,
                    "file_types": file_types,
                    "modified_directories": modified_directories,
                    "files": files  # Raw file data for further processing
                }
            return None
            
    except Exception as e:
        print(f"Error fetching commit details for {commit_sha}: {e}")
        return None

async def fetch_branch_stats(owner: str, repo: str, branch_name: str, token: str = None):
    """
    Lấy thống kê của branch (số commits, contributors)
    """
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"token {token}"
        elif GITHUB_TOKEN:
            headers["Authorization"] = f"token {GITHUB_TOKEN}"
        
        async with httpx.AsyncClient() as client:
            # Lấy số commits cho branch cụ thể
            commits_url = f"{BASE_URL}/repos/{owner}/{repo}/commits?sha={branch_name}&per_page=1"
            commits_response = await client.get(commits_url, headers=headers)
            commits_count = 0
            if commits_response.status_code == 200:
                # Đơn giản hóa: lấy từ response headers nếu có
                link_header = commits_response.headers.get("Link", "")
                if "rel=\"last\"" in link_header:
                    import re
                    last_page_match = re.search(r'page=(\d+)>; rel="last"', link_header)
                    if last_page_match:
                        commits_count = int(last_page_match.group(1))
            
            return {
                "commits_count": commits_count,
                "contributors_count": None  # Sẽ implement sau
            }
            
    except Exception as e:
        print(f"Error fetching branch stats: {e}")
        return {"commits_count": None, "contributors_count": None}

async def validate_github_token(token: str) -> bool:
    """
    Kiểm tra tính hợp lệ của GitHub token
    
    Args:
        token (str): GitHub token để kiểm tra
    
    Returns:
        bool: True nếu token hợp lệ
    """
    try:
        url = f"{BASE_URL}/user"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=get_headers(token))
            return response.status_code == 200
    except Exception:
        return False

async def get_rate_limit_info(token: str = None) -> Dict[str, Any]:
    """
    Lấy thông tin rate limit của GitHub API
    
    Args:
        token (str): GitHub token (optional)
    
    Returns:
        dict: Thông tin rate limit
    """
    try:
        url = f"{BASE_URL}/rate_limit"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=get_headers(token))
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        print(f"Error fetching rate limit: {e}")
    
    return {"resources": {"core": {"remaining": 0, "limit": 5000}}}

async def fetch_repository_languages(owner: str, repo: str, token: str = None) -> Dict[str, int]:
    """
    Lấy thông tin ngôn ngữ lập trình của repository
    
    Args:
        owner (str): Chủ sở hữu repo
        repo (str): Tên repository  
        token (str): GitHub token (optional)
    
    Returns:
        dict: Dictionary với key là ngôn ngữ, value là số bytes
    """
    try:
        url = f"/repos/{owner}/{repo}/languages"
        return await fetch_from_github(url, token)
    except Exception as e:
        print(f"Error fetching repository languages: {e}")
        return {}

async def fetch_repository_topics(owner: str, repo: str, token: str = None) -> List[str]:
    """
    Lấy danh sách topics của repository
    
    Args:
        owner (str): Chủ sở hữu repo
        repo (str): Tên repository
        token (str): GitHub token (optional)
    
    Returns:
        list: Danh sách topics
    """
    try:
        url = f"/repos/{owner}/{repo}/topics"
        headers = get_headers(token)
        headers["Accept"] = "application/vnd.github.mercy-preview+json"  # For topics API
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}{url}", headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data.get("names", [])
    except Exception as e:
        print(f"Error fetching repository topics: {e}")
    
    return []

async def fetch_branch_protection_rules(owner: str, repo: str, branch: str, token: str = None) -> Optional[Dict[str, Any]]:
    """
    Lấy thông tin protection rules của branch
    
    Args:
        owner (str): Chủ sở hữu repo
        repo (str): Tên repository
        branch (str): Tên branch
        token (str): GitHub token (optional)
    
    Returns:
        dict: Thông tin protection rules hoặc None nếu không có
    """
    try:
        url = f"/repos/{owner}/{repo}/branches/{branch}/protection"
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}{url}", headers=get_headers(token))
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None  # Branch not protected
    except Exception as e:
        print(f"Error fetching branch protection for {branch}: {e}")
    
    return None

async def fetch_enhanced_commits(
    owner: str, 
    repo: str, 
    branch: str = "main",
    token: str = None,
    since: str = None,
    until: str = None,
    per_page: int = 100
) -> List[Dict[str, Any]]:
    """
    Lấy commits với thông tin chi tiết bao gồm stats và files
    
    Args:
        owner (str): Chủ sở hữu repo
        repo (str): Tên repository
        branch (str): Tên branch
        token (str): GitHub token
        since (str): ISO datetime string
        until (str): ISO datetime string
        per_page (int): Số commits per page
    
    Returns:
        list: Danh sách commits với enhanced data
    """
    try:
        params = {
            "sha": branch,
            "per_page": min(per_page, 100)
        }
        
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        
        url = f"/repos/{owner}/{repo}/commits"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}{url}", 
                headers=get_headers(token),
                params=params
            )
            response.raise_for_status()
            commits_data = response.json()
        
        # Enrich each commit with detailed information
        enhanced_commits = []
        for commit in commits_data:
            sha = commit.get("sha")
            if not sha:
                continue
            
            # Get detailed commit info if needed
            enhanced_commit = {
                **commit,
                "enhanced_stats": None,
                "file_details": None
            }
            
            # Optionally fetch detailed stats for each commit
            # (This can be expensive for large repos)
            try:
                detailed_commit = await fetch_commit_details(sha, owner, repo, token)
                if detailed_commit:
                    enhanced_commit["enhanced_stats"] = detailed_commit.get("stats", {})
                    
            except Exception as e:
                print(f"Warning: Could not fetch enhanced data for commit {sha}: {e}")
            
            enhanced_commits.append(enhanced_commit)
        
        return enhanced_commits
        
    except Exception as e:
        print(f"Error fetching enhanced commits: {e}")
        return []

async def fetch_enhanced_commits_batch(
    owner: str, 
    repo: str, 
    branch: str = "main",
    token: str = None,
    since: str = None,
    until: str = None,
    max_commits: int = 100
) -> List[Dict[str, Any]]:
    """
    Lấy commits với thông tin chi tiết bao gồm enhanced metadata trong batch
    
    Args:
        owner (str): Chủ sở hữu repo
        repo (str): Tên repository
        branch (str): Tên branch
        token (str): GitHub token
        since (str): ISO datetime string
        until (str): ISO datetime string
        max_commits (int): Số commits tối đa
    
    Returns:
        list: Danh sách commits với enhanced metadata
    """
    try:
        # Get basic commits list first
        commits = await fetch_commits(
            token=token,
            owner=owner,
            name=repo,
            branch=branch,
            since=since,
            until=until,
            per_page=min(max_commits, 100),
            include_details=False
        )
        
        enhanced_commits = []
        
        # Process commits in smaller batches to avoid rate limits
        batch_size = 10
        for i in range(0, min(len(commits), max_commits), batch_size):
            batch = commits[i:i + batch_size]
            
            # Create tasks for concurrent fetching
            import asyncio
            tasks = []
            for commit in batch:
                commit_sha = commit.get("sha")
                if commit_sha:
                    task = fetch_commit_details(commit_sha, owner, repo, token)
                    tasks.append(task)
            
            # Execute batch concurrently
            if tasks:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        print(f"Error in batch processing: {result}")
                        continue
                    if result:
                        enhanced_commits.append(result)
            
            # Small delay to respect rate limits
            await asyncio.sleep(0.1)
        
        return enhanced_commits
        
    except Exception as e:
        print(f"Error fetching enhanced commits batch: {e}")
        return []

async def fetch_commit_files_metadata(
    owner: str,
    repo: str, 
    commit_sha: str,
    token: str = None
) -> Dict[str, Any]:
    """
    Lấy metadata chi tiết về files trong một commit
    
    Args:
        owner (str): Chủ sở hữu repo
        repo (str): Tên repository
        commit_sha (str): SHA của commit
        token (str): GitHub token
        
    Returns:
        dict: Metadata về files changed
    """
    try:
        url = f"/repos/{owner}/{repo}/commits/{commit_sha}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}{url}", headers=get_headers(token))
            
            if response.status_code == 200:
                commit_data = response.json()
                files = commit_data.get("files", [])
                
                # Analyze files
                file_analysis = {
                    "files_changed": len(files),
                    "modified_files": [],
                    "file_types": {},
                    "modified_directories": {},
                    "file_categories": {
                        "added": 0,
                        "modified": 0,
                        "deleted": 0,
                        "renamed": 0
                    },
                    "language_changes": {},
                    "size_changes": {
                        "additions": 0,
                        "deletions": 0,
                        "total": 0
                    }
                }
                
                for file_info in files:
                    filename = file_info.get("filename", "")
                    status = file_info.get("status", "")
                    additions = file_info.get("additions", 0)
                    deletions = file_info.get("deletions", 0)
                    
                    if filename:
                        file_analysis["modified_files"].append({
                            "filename": filename,
                            "status": status,
                            "additions": additions,
                            "deletions": deletions,
                            "changes": additions + deletions
                        })
                        
                        # File extension analysis
                        if "." in filename:
                            ext = "." + filename.split(".")[-1].lower()
                            if ext not in file_analysis["file_types"]:
                                file_analysis["file_types"][ext] = {
                                    "count": 0,
                                    "additions": 0,
                                    "deletions": 0
                                }
                            file_analysis["file_types"][ext]["count"] += 1
                            file_analysis["file_types"][ext]["additions"] += additions
                            file_analysis["file_types"][ext]["deletions"] += deletions
                        
                        # Directory analysis
                        if "/" in filename:
                            directory = "/".join(filename.split("/")[:-1])
                        else:
                            directory = "root"
                            
                        if directory not in file_analysis["modified_directories"]:
                            file_analysis["modified_directories"][directory] = {
                                "files": 0,
                                "additions": 0,
                                "deletions": 0
                            }
                        file_analysis["modified_directories"][directory]["files"] += 1
                        file_analysis["modified_directories"][directory]["additions"] += additions
                        file_analysis["modified_directories"][directory]["deletions"] += deletions
                        
                        # Status category
                        if status in file_analysis["file_categories"]:
                            file_analysis["file_categories"][status] += 1
                        
                        # Total size changes
                        file_analysis["size_changes"]["additions"] += additions
                        file_analysis["size_changes"]["deletions"] += deletions
                        file_analysis["size_changes"]["total"] += additions + deletions
                
                return file_analysis
                
    except Exception as e:
        print(f"Error fetching commit files metadata: {e}")
        
    return {
        "files_changed": 0,
        "modified_files": [],
        "file_types": {},
        "modified_directories": {},
        "file_categories": {},
        "size_changes": {"additions": 0, "deletions": 0, "total": 0}
    }

async def get_commit_comparison(
    owner: str,
    repo: str,
    base_sha: str,
    head_sha: str,
    token: str = None
) -> Dict[str, Any]:
    """
    So sánh 2 commits để lấy thông tin thay đổi
    
    Args:
        owner (str): Chủ sở hữu repo
        repo (str): Tên repository
        base_sha (str): SHA của commit gốc
        head_sha (str): SHA của commit đích
        token (str): GitHub token
        
    Returns:
        dict: Thông tin so sánh giữa 2 commits
    """
    try:
        url = f"/repos/{owner}/{repo}/compare/{base_sha}...{head_sha}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}{url}", headers=get_headers(token))
            
            if response.status_code == 200:
                comparison_data = response.json()
                
                return {
                    "status": comparison_data.get("status"),
                    "ahead_by": comparison_data.get("ahead_by"),
                    "behind_by": comparison_data.get("behind_by"),
                    "total_commits": comparison_data.get("total_commits"),
                    "commits": comparison_data.get("commits", []),
                    "files": comparison_data.get("files", []),
                    "stats": {
                        "additions": comparison_data.get("stats", {}).get("additions", 0),
                        "deletions": comparison_data.get("stats", {}).get("deletions", 0),
                        "total": comparison_data.get("stats", {}).get("total", 0)
                    }
                }
                
    except Exception as e:
        print(f"Error comparing commits: {e}")
        
    return {}