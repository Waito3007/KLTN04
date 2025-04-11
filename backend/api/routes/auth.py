# backend/api/routes/auth.py
from fastapi import APIRouter, Request
from core.oauth import oauth
import os

auth_router = APIRouter()

@auth_router.get("/login")
async def login(request: Request):
    redirect_uri = os.getenv("GITHUB_CALLBACK_URL")
    return await oauth.github.authorize_redirect(request, redirect_uri)

@auth_router.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.github.authorize_access_token(request)

    # Lấy thông tin user từ GitHub
    resp = await oauth.github.get("user", token=token)
    profile = resp.json()

    # Lấy email nếu chưa có trong profile
    if not profile.get("email"):
        emails_resp = await oauth.github.get("user/emails", token=token)
        emails = emails_resp.json()
        primary_email = next((e["email"] for e in emails if e["primary"]), None)
        profile["email"] = primary_email

    return {
        "token": token,
        "profile": profile
    }
