# backend/core/config.py
# File cấu hình chính cho ứng dụng FastAPI

# Import các thư viện cần thiết
import os  # Làm việc với biến môi trường
from fastapi.middleware.cors import CORSMiddleware  # Middleware CORS
from starlette.middleware.sessions import SessionMiddleware  # Middleware quản lý session
from fastapi import FastAPI  # Framework chính
from api.routes.sync import sync_router  # Router cho GitHub Sync API
from api.routes.repo import repo_router  # Router cho Repository API
from api.routes.auth import auth_router  # Router cho xác thực
from api.routes.commit import commit_router  # Router cho Commit API
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
            "http://localhost:3000",  # Frontend dev (React có thể chạy ở port 3000)
            "http://127.0.0.1:5173",  # Alternative localhost
            "http://127.0.0.1:3000",  # Alternative localhost
            "*"  # Allow all origins for development (remove in production)
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
    
    # Đăng ký GitHub sync router với prefix /api
    app.include_router(sync_router, prefix="/api")
    
    # Đăng ký repository router với prefix /api
    app.include_router(repo_router, prefix="/api")
    
    # Đăng ký commit router với prefix /api/commits
    app.include_router(commit_router, prefix="/api")