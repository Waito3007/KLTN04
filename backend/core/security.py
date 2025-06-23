# backend/core/security.py
"""
Security module for authentication and authorization
Handles GitHub OAuth tokens and user session management
"""

from fastapi import Depends, HTTPException, status, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.utils import get_authorization_scheme_param
from fastapi.security.base import SecurityBase
from fastapi.openapi.models import HTTPBearer as HTTPBearerModel
from starlette.requests import Request
from typing import Optional, Dict, Any
import httpx
import logging
from functools import lru_cache

from services.user_service import get_user_by_github_id
from db.models.users import users
from db.database import database, engine
from sqlalchemy import select

logger = logging.getLogger(__name__)

class GitHubTokenBearer(SecurityBase):
    """
    Custom security scheme that accepts both 'Bearer' and 'token' schemes
    """
    def __init__(self, auto_error: bool = True):
        self.auto_error = auto_error

    async def __call__(self, request: Request) -> Optional[str]:
        authorization = request.headers.get("Authorization")
        if not authorization:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return None

        try:
            scheme, credentials = authorization.split(' ', 1)
            if scheme.lower() not in ["bearer", "token"]:
                if self.auto_error:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid authentication credentials",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                return None
        except ValueError:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authorization header format",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return None

        return credentials  # Return just the token string

# Security scheme for both Bearer and token formats
security = GitHubTokenBearer(auto_error=False)

class CurrentUser:
    """Current user data structure"""
    def __init__(self, user_data: dict):
        self.id = user_data.get("id")
        self.github_id = user_data.get("github_id")
        self.github_username = user_data.get("github_username")
        self.email = user_data.get("email")
        self.display_name = user_data.get("display_name")
        self.full_name = user_data.get("full_name")
        self.avatar_url = user_data.get("avatar_url")
        self.is_active = user_data.get("is_active", True)
        self.is_verified = user_data.get("is_verified", False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "github_id": self.github_id,
            "github_username": self.github_username,
            "username": self.github_username,  # Alias for backward compatibility
            "email": self.email,
            "display_name": self.display_name,
            "full_name": self.full_name,
            "avatar_url": self.avatar_url,
            "is_active": self.is_active,
            "is_verified": self.is_verified
        }

async def verify_github_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify GitHub token and get user info
    Note: LRU cache removed as it doesn't work with async functions
    """
    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Get user info from GitHub API
            response = await client.get("https://api.github.com/user", headers=headers)
            
            if response.status_code == 200:
                github_user = response.json()
                return github_user
            elif response.status_code == 401:
                logger.warning("Invalid GitHub token provided")
                return None
            else:
                logger.error(f"GitHub API error: {response.status_code}")
                return None
                
    except Exception as e:
        logger.error(f"Error verifying GitHub token: {e}")
        return None

async def get_current_user_from_token(token: str) -> Optional[CurrentUser]:
    """
    Get current user from GitHub token
    """
    try:
        # Verify token with GitHub
        github_user = await verify_github_token(token)
        if not github_user:
            return None
        
        # Get user from our database using engine connection
        with engine.connect() as conn:
            query = select(users).where(users.c.github_id == github_user["id"])
            db_user = conn.execute(query).fetchone()
            
            if not db_user:
                logger.warning(f"User {github_user['login']} not found in database")
                return None
            
            # Check if user is active (handle None values properly)
            if db_user.is_active is False:  # Only reject if explicitly False
                logger.warning(f"User {github_user['login']} is inactive")
                return None
            
            # Convert row to dict using _mapping
            user_dict = dict(db_user._mapping)
            return CurrentUser(user_dict)
        
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        return None

async def get_current_user(
    token: Optional[str] = Depends(security)
) -> CurrentUser:
    """
    FastAPI dependency to get current authenticated user
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = await get_current_user_from_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

async def get_current_user_optional(
    token: Optional[str] = Depends(security)
) -> Optional[CurrentUser]:
    """
    FastAPI dependency to get current user (optional)
    Returns None if not authenticated instead of raising error
    """
    if not token:
        return None
    
    return await get_current_user_from_token(token)

# Alternative dependency that accepts token from header (supports both Bearer and token formats)
async def get_current_user_from_header(
    authorization: Optional[str] = Header(None)
) -> CurrentUser:
    """
    FastAPI dependency to get current user from Authorization header
    Supports both "Bearer <token>" and "token <token>" formats
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
        )
    
    # Extract token from "Bearer <token>" or "token <token>" format
    try:
        scheme, token = authorization.split(' ', 1)  # Split only on first space
        if scheme.lower() not in ["bearer", "token"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme. Use 'Bearer' or 'token'",
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
        )
    
    user = await get_current_user_from_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    
    return user

# thêm một hàm để lấy người dùng hiện tại với tùy chọn trả về None nếu không có token
async def get_current_user_strict_optional(
    request: Request
) -> Optional[CurrentUser]:
    """
    FastAPI dependency to get current user with strict validation
    - If no Authorization header: returns None (allowed)
    - If Authorization header exists but invalid: raises 401 error
    - If Authorization header exists and valid: returns CurrentUser
    """
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None  # No token provided - this is OK
    
    try:
        scheme, credentials = authorization.split(' ', 1)
        if scheme.lower() not in ["bearer", "token"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme. Use 'Bearer' or 'token'",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Token provided - must be valid
    user = await get_current_user_from_token(credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

# Convenience function for backward compatibility
async def get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Legacy function for backward compatibility
    Returns user dict instead of CurrentUser object
    """
    user = await get_current_user_from_token(token)
    return user.to_dict() if user else None
