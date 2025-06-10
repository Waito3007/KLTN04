# backend/services/github_service.py
# Service xử lý các tương tác với GitHub API

# Import các thư viện cần thiết
import httpx  # Thư viện HTTP client async
import os  # Làm việc với biến môi trường
from dotenv import load_dotenv  # Đọc file .env
from typing import Optional  # Để khai báo kiểu dữ liệu optional
load_dotenv()  # Nạp biến môi trường từ file .env

# Lấy GitHub token từ biến môi trường
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Base URL cho GitHub API
BASE_URL = "https://api.github.com"

# Headers mặc định cho các request GitHub API
headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",  # Token xác thực
    "Accept": "application/vnd.github+json",  # Loại response mong muốn
}

async def fetch_from_github(url: str):
    """
    Hàm tổng quát để fetch dữ liệu từ GitHub API
    
    Args:
        url (str): Phần cuối của URL (sau BASE_URL)
    
    Returns:
        dict: Dữ liệu JSON trả về từ GitHub API
    
    Raises:
        HTTPError: Nếu request lỗi
    """
    async with httpx.AsyncClient() as client:
        # Gọi GET request tới GitHub API
        response = await client.get(f"{BASE_URL}{url}", headers=headers)
        # Tự động raise exception nếu có lỗi HTTP
        response.raise_for_status()
        # Trả về dữ liệu dạng JSON
        return response.json()

async def fetch_commits(
    token: str, 
    owner: str, 
    name: str, 
    branch: str, 
    since: Optional[str], 
    until: Optional[str]
):
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
    # Xây dựng URL API để lấy commit
    url = f"https://api.github.com/repos/{owner}/{name}/commits"
    
    # Headers cho request
    headers = {
        "Authorization": f"token {token}",  # Sử dụng token từ tham số
        "Accept": "application/vnd.github+json"  # Loại response mong muốn
    }
    
    # Parameters cho request
    params = {
        "sha": branch  # Lọc theo branch
    }
    
    # Thêm tham số lọc thời gian nếu có
    if since:
        params["since"] = since
    if until:
        params["until"] = until

    # Gọi API GitHub
    async with httpx.AsyncClient() as client:
        res = await client.get(url, headers=headers, params=params)
        # Kiểm tra lỗi HTTP
        res.raise_for_status()
        # Trả về dữ liệu dạng JSON
        return res.json()