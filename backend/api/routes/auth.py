# KLTN04\backend\api\routes\auth.py
# Import các thư viện cần thiết
from fastapi import APIRouter, Request  # APIRouter để tạo routes, Request để xử lý request HTTP
from core.oauth import oauth  # Module tự định nghĩa chứa cấu hình OAuth
from fastapi.responses import RedirectResponse  # Để thực hiện chuyển hướng
import os  # Để làm việc với biến môi trường

# Khởi tạo router cho các endpoint xác thực
auth_router = APIRouter()

# Endpoint /login để bắt đầu quá trình xác thực với GitHub
@auth_router.get("/login")
async def login(request: Request):
    # Lấy callback URL từ biến môi trường
    redirect_uri = os.getenv("GITHUB_CALLBACK_URL")
    
    # Chuyển hướng người dùng đến trang xác thực GitHub
    return await oauth.github.authorize_redirect(request, redirect_uri)

# Endpoint /auth/callback - GitHub sẽ gọi lại endpoint này sau khi xác thực thành công
@auth_router.get("/auth/callback")
async def auth_callback(request: Request):
    # Lấy access token từ response của GitHub
    token = await oauth.github.authorize_access_token(request)
    
    # Gọi API GitHub để lấy thông tin user cơ bản
    resp = await oauth.github.get("user", token=token)
    profile = resp.json()  # Chuyển response thành dictionary

    # Xử lý trường hợp profile không có email
    if not profile.get("email"):
        # Gọi API riêng để lấy danh sách email
        emails_resp = await oauth.github.get("user/emails", token=token)
        emails = emails_resp.json()
        
        # Tìm email được đánh dấu là primary (chính)
        primary_email = next((e["email"] for e in emails if e["primary"]), None)
        
        # Gán email chính vào profile
        profile["email"] = primary_email

    # Tạo URL để chuyển hướng về frontend (React) với các thông tin user
    redirect_url = (
        f"http://localhost:5173/api/auth-success"  # URL frontend
        f"?token={token['access_token']}"          # Access token
        f"&username={profile['login']}"           # Tên đăng nhập GitHub
        f"&email={profile['email']}"              # Email user
        f"&avatar_url={profile['avatar_url']}"    # URL avatar
    )

    # Thực hiện chuyển hướng về frontend
    return RedirectResponse(redirect_url)