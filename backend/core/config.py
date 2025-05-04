# backend/core/config.py
import os
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi import FastAPI
from api.routes.github import github_router
from api.routes.auth import auth_router
from dotenv import load_dotenv

load_dotenv()

def setup_middlewares(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",  # frontend dev
            "http://localhost:3000"   # thêm nếu dùng port 3000
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],  # 👈 Quan trọng để hỗ trợ Authorization
    )

    app.add_middleware(
        SessionMiddleware,
        secret_key=os.getenv('SECRET_KEY')
    )


def setup_routers(app: FastAPI):
    app.include_router(auth_router, prefix="/auth")
    app.include_router(github_router, prefix="/api")  # <- Gộp lại, không bị đè
