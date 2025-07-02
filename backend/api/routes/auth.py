# KLTN04\backend\api\routes\auth.py
from fastapi import APIRouter, Request, HTTPException, Depends
from core.oauth import oauth
from fastapi.responses import RedirectResponse
from services.user_service import save_user  # Import hàm lưu người dùng
from core.security import get_current_user, get_current_user_optional, CurrentUser
from utils.auth_utils import (
    validate_github_profile, 
    sanitize_user_data, 
    create_auth_response,
    AuthenticationError
)
import os
from datetime import datetime

auth_router = APIRouter()

# Endpoint /login để bắt đầu quá trình xác thực với GitHub
@auth_router.get("/login")
async def login(request: Request):
    try:
        # Lấy callback URL từ biến môi trường
        redirect_uri = os.getenv("GITHUB_CALLBACK_URL")
        
        if not redirect_uri:
            raise HTTPException(
                status_code=500, 
                detail="GitHub callback URL not configured"
            )
        
        # Chuyển hướng người dùng đến trang xác thực GitHub
        return await oauth.github.authorize_redirect(request, redirect_uri)
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Login error: {e}")
        print(f"Full traceback: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Login initialization failed: {str(e)}"
        )

# Endpoint /auth/callback - GitHub sẽ gọi lại endpoint này sau khi xác thực thành công
@auth_router.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        # Kiểm tra mã code từ GitHub
        code = request.query_params.get("code")
        if not code:
            raise HTTPException(
                status_code=400, 
                detail="Authorization code missing from GitHub response"
            )
        
        # Kiểm tra lỗi từ GitHub
        error = request.query_params.get("error")
        if error:
            error_description = request.query_params.get("error_description", "Unknown error")
            raise HTTPException(
                status_code=400,
                detail=f"GitHub authorization error: {error} - {error_description}"
            )
        
        # Lấy access token từ GitHub
        try:
            token = await oauth.github.authorize_access_token(request)
            if not token or not token.get("access_token"):
                raise HTTPException(
                    status_code=400, 
                    detail="Failed to obtain access token from GitHub"
                )
        except Exception as token_error:
            print(f"Failed to get access token: {token_error}")
            raise HTTPException(
                status_code=400, 
                detail="Invalid authorization code or GitHub API error"
            )
        
        # Gọi API GitHub để lấy thông tin user cơ bản
        try:
            resp = await oauth.github.get("user", token=token)
            profile = resp.json()  # Chuyển response thành dictionary
        except Exception as profile_error:
            print(f"Failed to get user profile: {profile_error}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve user profile from GitHub"
            )

        # Lấy email nếu không có trong profile
        if not profile.get("email"):
            try:
                # Gọi API riêng để lấy danh sách email
                emails_resp = await oauth.github.get("user/emails", token=token)
                emails = emails_resp.json()
                
                # Tìm email được đánh dấu là primary (chính)
                primary_email = next((e["email"] for e in emails if e["primary"]), None)
                
                # Gán email chính vào profile
                profile["email"] = primary_email
            except Exception as email_error:
                print(f"Failed to get user email: {email_error}")
                # Continue without email - some users may have private emails

        # Kiểm tra thông tin bắt buộc
        if not profile.get("login"):
            raise HTTPException(
                status_code=400, 
                detail="GitHub username is required but not provided"
            )
        
        if not profile.get("email"):
            print(f"Warning: No email found for user {profile.get('login')}")
            # Set a default email pattern for users with private emails
            profile["email"] = f"{profile['login']}@users.noreply.github.com"
          # Lưu thông tin người dùng vào cơ sở dữ liệu
        # Parse github_created_at to handle timezone properly
        github_created_at = None
        if profile.get("created_at"):
            try:
                from datetime import datetime
                import dateutil.parser
                github_created_at = dateutil.parser.parse(profile["created_at"]).replace(tzinfo=None)
            except Exception as date_error:
                print(f"Error parsing github_created_at: {date_error}")
                github_created_at = None
        
        user_data = {
            "github_id": profile["id"],
            "github_username": profile["login"],
            "email": profile["email"],
            "display_name": profile.get("name"),  # GitHub display name
            "full_name": profile.get("name"),     # Same as display name
            "avatar_url": profile.get("avatar_url"),
            "bio": profile.get("bio"),
            "location": profile.get("location"),
            "company": profile.get("company"),
            "blog": profile.get("blog"),
            "twitter_username": profile.get("twitter_username"),
            "github_profile_url": profile.get("html_url"),
            "repos_url": profile.get("repos_url"),
            "github_created_at": github_created_at,
            # Set default active status
            "is_active": True,
            "is_verified": False
        }
        
        # Lưu user data với error handling
        try:
            await save_user(user_data)
        except Exception as save_error:
            print(f"Error saving user data: {save_error}")
            raise HTTPException(
                status_code=500,
                detail="Failed to save user information to database"
            )

        # Validate redirect URL components
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
        if not frontend_url:
            raise HTTPException(
                status_code=500,
                detail="Frontend URL not configured"
            )

        # Redirect về frontend với token và thông tin người dùng
        redirect_url = (
            f"{frontend_url}/auth-success"
            f"?token={token['access_token']}"
            f"&username={profile['login']}"
            f"&email={profile['email']}"
            f"&avatar_url={profile.get('avatar_url', '')}"
        )

        # Thực hiện chuyển hướng về frontend
        return RedirectResponse(redirect_url)
        
    except HTTPException:
        # Re-raise HTTP exceptions (they're already properly formatted)
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Auth callback error: {e}")
        print(f"Full traceback: {error_details}")
        
        # Redirect to frontend with error information
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
        error_redirect_url = f"{frontend_url}/auth-error?error={str(e)}"
        return RedirectResponse(error_redirect_url)

# Thêm endpoint để kiểm tra thông tin user hiện tại
@auth_router.get("/me")
async def get_current_user_info(current_user: CurrentUser = Depends(get_current_user)):
    """
    Get current authenticated user information
    """
    return {
        "success": True,
        "user": current_user.to_dict(),
        "message": "User authenticated successfully"
    }

@auth_router.get("/me/optional")
async def get_current_user_optional_info(current_user: CurrentUser = Depends(get_current_user_optional)):
    """
    Get current user info (optional authentication)
    """
    if current_user:
        return {
            "authenticated": True,
            "user": current_user.to_dict()
        }
    else:
        return {
            "authenticated": False,
            "user": None
        }