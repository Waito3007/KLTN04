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
    until: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Lấy danh sách commit từ repository GitHub
    
    Args:
        token (str): GitHub access token
        owner (str): Chủ repository
        name (str): Tên repository
        branch (str): Tên branch
        since (Optional[str]): Lọc commit từ thời gian này (ISO format)
        until (Optional[str]): Lọc commit đến thời gian này (ISO format)
    
    Returns:
        list: Danh sách commit
    
    Raises:
        HTTPError: Nếu request lỗi
    """
    url = f"/repos/{owner}/{name}/commits"
    
    # Parameters cho request
    params = {"sha": branch}
    
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
        return response.json()

async def fetch_commit_details(commit_sha: str, owner: str, repo: str, token: str = None) -> Optional[Dict[str, Any]]:
    """
    Lấy thông tin chi tiết của một commit
    
    Args:
        commit_sha (str): SHA hash của commit
        owner (str): Chủ sở hữu repo
        repo (str): Tên repository
        token (str): GitHub token (optional)
    
    Returns:
        dict: Thông tin chi tiết commit hoặc None nếu lỗi
    """
    try:
        url = f"/repos/{owner}/{repo}/commits/{commit_sha}"
        
        async with httpx.AsyncClient() as client:
            full_url = f"{BASE_URL}{url}"
            response = await client.get(full_url, headers=get_headers(token))
            
            if response.status_code == 200:
                commit_data = response.json()
                return {
                    "sha": commit_data.get("sha"),
                    "date": commit_data.get("commit", {}).get("committer", {}).get("date"),
                    "author_name": commit_data.get("commit", {}).get("author", {}).get("name"),
                    "author_email": commit_data.get("commit", {}).get("author", {}).get("email"),
                    "committer_name": commit_data.get("commit", {}).get("committer", {}).get("name"),
                    "committer_email": commit_data.get("commit", {}).get("committer", {}).get("email"),
                    "message": commit_data.get("commit", {}).get("message"),
                    "url": commit_data.get("html_url"),
                    "stats": {
                        "additions": commit_data.get("stats", {}).get("additions", 0),
                        "deletions": commit_data.get("stats", {}).get("deletions", 0),
                        "total": commit_data.get("stats", {}).get("total", 0)
                    }
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

async def fetch_commit_files(commit_sha: str, owner: str, repo: str, token: str = None) -> List[Dict[str, Any]]:
    """
    Lấy danh sách files thay đổi trong một commit
    
    Args:
        commit_sha (str): SHA của commit
        owner (str): Chủ sở hữu repo
        repo (str): Tên repository
        token (str): GitHub token
    
    Returns:
        list: Danh sách files với thông tin thay đổi
    """
    try:
        url = f"/repos/{owner}/{repo}/commits/{commit_sha}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}{url}", headers=get_headers(token))
            
            if response.status_code == 200:
                commit_data = response.json()
                files = commit_data.get("files", [])
                
                # Process file information
                processed_files = []
                for file in files:
                    processed_files.append({
                        "filename": file.get("filename"),
                        "status": file.get("status"),  # added, modified, removed, renamed
                        "additions": file.get("additions", 0),
                        "deletions": file.get("deletions", 0),
                        "changes": file.get("changes", 0),
                        "patch": file.get("patch"),  # Actual diff content
                        "previous_filename": file.get("previous_filename"),  # For renamed files
                        "blob_url": file.get("blob_url"),
                        "raw_url": file.get("raw_url")
                    })
                
                return processed_files
            
            return []
            
    except Exception as e:
        print(f"Error fetching commit files for {commit_sha}: {e}")
        return []

async def fetch_commit_author_info(commit_sha: str, owner: str, repo: str, token: str = None) -> Optional[Dict[str, Any]]:
    """
    Lấy thông tin chi tiết về author của commit (bao gồm GitHub user info nếu có)
    
    Args:
        commit_sha (str): SHA của commit
        owner (str): Chủ sở hữu repo
        repo (str): Tên repository
        token (str): GitHub token
    
    Returns:
        dict: Thông tin author chi tiết
    """
    try:
        url = f"/repos/{owner}/{repo}/commits/{commit_sha}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}{url}", headers=get_headers(token))
            
            if response.status_code == 200:
                commit_data = response.json()
                
                # Extract author information
                author_info = {
                    "git_author": commit_data.get("commit", {}).get("author", {}),
                    "git_committer": commit_data.get("commit", {}).get("committer", {}),
                    "github_author": commit_data.get("author"),  # GitHub user info
                    "github_committer": commit_data.get("committer")  # GitHub user info
                }
                
                return author_info
            
            return None
            
    except Exception as e:
        print(f"Error fetching author info for {commit_sha}: {e}")
        return None

async def analyze_commit_type(message: str, files: List[Dict] = None) -> Dict[str, Any]:
    """
    Phân tích loại commit dựa trên message và files thay đổi
    
    Args:
        message (str): Commit message
        files (list): Danh sách files thay đổi
    
    Returns:
        dict: Thông tin phân loại commit
    """
    import re
    
    analysis = {
        "type": "other",
        "is_feature": False,
        "is_bugfix": False,
        "is_refactor": False,
        "is_documentation": False,
        "is_test": False,
        "is_merge": False,
        "conventional_commit": False,
        "breaking_change": False,
        "scope": None
    }
    
    message_lower = message.lower()
    
    # Check for merge commits
    if message.startswith("Merge") or "merge" in message_lower:
        analysis["is_merge"] = True
        analysis["type"] = "merge"
    
    # Check for conventional commits (feat:, fix:, docs:, etc.)
    conventional_pattern = r"^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+\))?\!?:"
    match = re.match(conventional_pattern, message)
    if match:
        analysis["conventional_commit"] = True
        commit_type = match.group(1)
        scope = match.group(2)
        
        analysis["type"] = commit_type
        if scope:
            analysis["scope"] = scope.strip("()")
        
        # Check for breaking changes
        if "!" in match.group(0) or "BREAKING CHANGE" in message:
            analysis["breaking_change"] = True
        
        # Set specific flags
        analysis["is_feature"] = commit_type == "feat"
        analysis["is_bugfix"] = commit_type == "fix"
        analysis["is_refactor"] = commit_type == "refactor"
        analysis["is_documentation"] = commit_type == "docs"
        analysis["is_test"] = commit_type == "test"
    
    else:
        # Fallback analysis based on keywords
        if any(keyword in message_lower for keyword in ["add", "implement", "feature", "new"]):
            analysis["is_feature"] = True
            analysis["type"] = "feature"
        elif any(keyword in message_lower for keyword in ["fix", "bug", "issue", "error"]):
            analysis["is_bugfix"] = True
            analysis["type"] = "bugfix"
        elif any(keyword in message_lower for keyword in ["refactor", "restructure", "improve"]):
            analysis["is_refactor"] = True
            analysis["type"] = "refactor"
        elif any(keyword in message_lower for keyword in ["doc", "readme", "comment"]):
            analysis["is_documentation"] = True
            analysis["type"] = "documentation"
        elif any(keyword in message_lower for keyword in ["test", "spec", "coverage"]):
            analysis["is_test"] = True
            analysis["type"] = "test"
    
    # Analyze files if provided
    if files:
        file_types = []
        for file in files:
            filename = file.get("filename", "")
            if filename.endswith((".md", ".txt", ".rst")):
                file_types.append("documentation")
            elif filename.endswith((".test.", ".spec.", "_test.", "_spec.")):
                file_types.append("test")
            elif filename.endswith((".py", ".js", ".ts", ".java", ".cpp", ".c")):
                file_types.append("code")
            elif filename.endswith((".css", ".scss", ".less")):
                file_types.append("style")
            elif filename.endswith((".json", ".yml", ".yaml", ".toml", ".ini")):
                file_types.append("config")
        
        # Update analysis based on file types
        if "documentation" in file_types and len(set(file_types)) == 1:
            analysis["is_documentation"] = True
            if analysis["type"] == "other":
                analysis["type"] = "documentation"
        
        if "test" in file_types and len(set(file_types)) == 1:
            analysis["is_test"] = True
            if analysis["type"] == "other":
                analysis["type"] = "test"
    
    return analysis