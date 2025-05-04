from fastapi import APIRouter, Request, HTTPException
from core.oauth import oauth
from fastapi.responses import RedirectResponse
from services.user_service import save_user  # Import hàm lưu người dùng
import os

auth_router = APIRouter()

@auth_router.get("/login")
async def login(request: Request):
    redirect_uri = os.getenv("GITHUB_CALLBACK_URL")
    return await oauth.github.authorize_redirect(request, redirect_uri)

@auth_router.get("/auth/callback")
async def auth_callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing code")

    # Lấy access token từ GitHub
    token = await oauth.github.authorize_access_token(request)
    resp = await oauth.github.get("user", token=token)
    profile = resp.json()

    # Lấy email nếu không có trong profile
    if not profile.get("email"):
        emails_resp = await oauth.github.get("user/emails", token=token)
        emails = emails_resp.json()
        primary_email = next((e["email"] for e in emails if e["primary"]), None)
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

    return RedirectResponse(redirect_url)
