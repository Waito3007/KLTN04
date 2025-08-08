# backend/utils/auth_utils.py
"""
Authentication utility functions for improved error handling and validation
"""

import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException
import httpx

logger = logging.getLogger(__name__)

class AuthenticationError(Exception):
    """Custom exception for authentication errors"""
    def __init__(self, message: str, status_code: int = 401):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

def validate_github_profile(profile: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate and sanitize GitHub profile data
    
    Args:
        profile: Raw GitHub profile data
        
    Returns:
        Dict containing validation errors (empty if valid)
    """
    errors = {}
    
    if not profile.get("login"):
        errors["username"] = "GitHub username is required"
    
    if not profile.get("id"):
        errors["github_id"] = "GitHub ID is required"
    
    # Email validation (allow fallback for private emails)
    if not profile.get("email"):
        logger.warning(f"No email found for user {profile.get('login')}")
    
    return errors

def create_fallback_email(username: str) -> str:
    """
    Create a fallback email for users with private email settings
    
    Args:
        username: GitHub username
        
    Returns:
        Fallback email address
    """
    return f"{username}@users.noreply.github.com"

async def validate_github_token(token: str) -> bool:
    """
    Validate GitHub token by making a test API call
    
    Args:
        token: GitHub access token
        
    Returns:
        True if token is valid, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = await client.get("https://api.github.com/user", headers=headers)
            return response.status_code == 200
            
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return False

def sanitize_user_data(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize and prepare user data for database storage
    
    Args:
        profile: GitHub profile data
        
    Returns:
        Sanitized user data dictionary
    """
    # Handle github_created_at parsing
    github_created_at = None
    if profile.get("created_at"):
        try:
            from datetime import datetime
            import dateutil.parser
            github_created_at = dateutil.parser.parse(profile["created_at"]).replace(tzinfo=None)
        except Exception as date_error:
            logger.error(f"Error parsing github_created_at: {date_error}")
            github_created_at = None
    
    # Ensure email exists (use fallback if needed)
    email = profile.get("email")
    if not email:
        email = create_fallback_email(profile["login"])
    
    return {
        "github_id": profile["id"],
        "github_username": profile["login"],
        "email": email,
        "display_name": profile.get("name"),
        "full_name": profile.get("name"),
        "avatar_url": profile.get("avatar_url"),
        "bio": profile.get("bio"),
        "location": profile.get("location"),
        "company": profile.get("company"),
        "blog": profile.get("blog"),
        "twitter_username": profile.get("twitter_username"),
        "github_profile_url": profile.get("html_url"),
        "repos_url": profile.get("repos_url"),
        "github_created_at": github_created_at,
        "is_active": True,
        "is_verified": False
    }

def create_auth_response(success: bool, message: str, user_data: Optional[Dict] = None, error_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Create standardized authentication response
    
    Args:
        success: Whether authentication was successful
        message: Response message
        user_data: User data (if successful)
        error_code: Error code (if failed)
        
    Returns:
        Standardized response dictionary
    """
    response = {
        "success": success,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if success and user_data:
        response["user"] = user_data
    elif not success and error_code:
        response["error_code"] = error_code
    
    return response
