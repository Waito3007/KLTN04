# backend/core/config.py
# File cấu hình chính cho ứng dụng FastAPI

# Import các thư viện cần thiết
import os  # Làm việc với biến môi trường
from fastapi.middleware.cors import CORSMiddleware  # Middleware CORS
from starlette.middleware.sessions import SessionMiddleware  # Middleware quản lý session
from fastapi import FastAPI  # Framework chính
from api.routes.github import github_router  # Router cho GitHub API
from api.routes.auth import auth_router  # Router cho xác thực
from dotenv import load_dotenv  # Đọc file .env

# Nạp biến môi trường từ file .env
load_dotenv()

# Hàm cấu hình các middleware cho ứng dụng
def setup_middlewares(app: FastAPI):
    """
    Thiết lập các middleware cần thiết cho ứng dụng
    
    Args:
        app (FastAPI): Instance của FastAPI app
    """
    
    # Thêm middleware CORS (Cross-Origin Resource Sharing)
    app.add_middleware(
        CORSMiddleware,
        # Danh sách domain được phép truy cập
        allow_origins=[
            "http://localhost:5173",  # Frontend dev (Vite thường chạy ở port 5173)
            "http://localhost:3000"   # Frontend dev (React có thể chạy ở port 3000)
        ],
        allow_credentials=True,  # Cho phép gửi credential (cookies, auth headers)
        allow_methods=["*"],  # Cho phép tất cả HTTP methods
        allow_headers=["*"],  # Cho phép tất cả headers (bao gồm Authorization)
    )

    # Thêm middleware quản lý session
    app.add_middleware(
        SessionMiddleware,
        secret_key=os.getenv('SECRET_KEY')  # Khóa bí mật từ biến môi trường
    )


# Hàm cấu hình các router cho ứng dụng
def setup_routers(app: FastAPI):
    """
    Đăng ký các router chính của ứng dụng
    
    Args:
        app (FastAPI): Instance của FastAPI app
    """
    
    # Đăng ký auth router với prefix /auth
    app.include_router(auth_router, prefix="/auth")
    
    # Đăng ký github router với prefix /api
    app.include_router(github_router, prefix="/api")  # Gộp chung không bị đè lẫn nhau