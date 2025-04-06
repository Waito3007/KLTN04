from fastapi import APIRouter, Request
from authlib.integrations.starlette_client import OAuth
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Tạo một router cho phần authentication
auth_router = APIRouter()

# Tạo instance OAuth
oauth = OAuth()

# GitHub OAuth configuration
oauth.register(
    name='github',
    client_id=os.getenv('GITHUB_CLIENT_ID'),
    client_secret=os.getenv('GITHUB_CLIENT_SECRET'),
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    access_token_url='https://github.com/login/oauth/access_token',
    refresh_token_url=None,
    client_kwargs={'scope': 'user:email'},
)

@auth_router.get('/login')
async def login(request: Request):
    redirect_uri = os.getenv('GITHUB_CALLBACK_URL')
    return await oauth.github.authorize_redirect(request, redirect_uri)

@auth_router.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.github.authorize_access_token(request)
    user = await oauth.github.parse_id_token(request, token)
    return {"user": user}
