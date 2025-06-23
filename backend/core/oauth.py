# backend/core/oauth.py
# File cấu hình OAuth cho ứng dụng, chủ yếu dùng cho GitHub OAuth

# Import các thư viện cần thiết
import os  # Để làm việc với biến môi trường
from dotenv import load_dotenv  # Để đọc file .env
from authlib.integrations.starlette_client import OAuth  # Thư viện OAuth cho Starlette/FastAPI

# Load các biến môi trường từ file .env
load_dotenv()

# Khởi tạo instance OAuth
oauth = OAuth()

# Đăng ký provider GitHub cho OAuth
oauth.register(
    name='github',  # Tên provider
    
    # Client ID từ ứng dụng GitHub OAuth App
    client_id=os.getenv('GITHUB_CLIENT_ID'),
    
    # Client Secret từ ứng dụng GitHub OAuth App
    client_secret=os.getenv('GITHUB_CLIENT_SECRET'),
    
    # URL để lấy access token
    access_token_url='https://github.com/login/oauth/access_token',
    
    # Các params thêm khi lấy access token (None nếu không có)
    access_token_params=None,
    
    # URL để xác thực
    authorize_url='https://github.com/login/oauth/authorize',
    
    # Các params thêm khi xác thực (None nếu không có)
    authorize_params=None,
    
    # Base URL cho API GitHub
    api_base_url='https://api.github.com/',
    
    # Các tham số bổ sung cho client
    client_kwargs={
        'scope': 'read:user user:email repo'  # Các quyền yêu cầu
        # read:user - Đọc thông tin user
        # user:email - Đọc email user
        # repo - Truy cập repository
    }
)