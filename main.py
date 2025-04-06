from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from auth import auth_router
import os
from dotenv import load_dotenv

# Load .env file để lấy SECRET_KEY
load_dotenv()

# Tạo instance FastAPI
app = FastAPI()

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Địa chỉ frontend React của bạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thêm SessionMiddleware để quản lý phiên làm việc
app.add_middleware(SessionMiddleware, secret_key=os.getenv('SECRET_KEY'))

# Đăng ký router auth từ file auth.py
app.include_router(auth_router)
