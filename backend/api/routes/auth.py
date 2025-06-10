# KLTN04\backend\api\routes\auth.py
from fastapi import APIRouter, Request, HTTPException
from core.oauth import oauth
from fastapi.responses import RedirectResponse
from services.user_service import save_user  # Import hàm lưu người dùng
import os

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
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing code")

    # Lấy access token từ GitHub
    token = await oauth.github.authorize_access_token(request)
    
    # Gọi API GitHub để lấy thông tin user cơ bản
    resp = await oauth.github.get("user", token=token)
    profile = resp.json()  # Chuyển response thành dictionary

    # Lấy email nếu không có trong profile
    if not profile.get("email"):
        # Gọi API riêng để lấy danh sách email
        emails_resp = await oauth.github.get("user/emails", token=token)
        emails = emails_resp.json()
        
        # Tìm email được đánh dấu là primary (chính)
        primary_email = next((e["email"] for e in emails if e["primary"]), None)
        
        # Gán email chính vào profile
        profile["email"] = primary_email

    # Kiểm tra thông tin bắt buộc
    if not profile.get("email") or not profile.get("login"):
        raise HTTPException(status_code=400, detail="Missing required user information")

    # Lưu thông tin người dùng vào cơ sở dữ liệu
    user_data = {
        "github_id": profile["id"],
        "github_username": profile["login"],
        "email": profile["email"],
        "avatar_url": profile["avatar_url"],
    }
    await save_user(user_data)

    # Redirect về frontend với token và thông tin người dùng
    redirect_url = (
        f"http://localhost:5173/auth-success"
        f"?token={token['access_token']}"
        f"&username={profile['login']}"
        f"&email={profile['email']}"
        f"&avatar_url={profile['avatar_url']}"
    )

    # Thực hiện chuyển hướng về frontend
    return RedirectResponse(redirect_url)