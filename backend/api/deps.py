# KLTN04\backend\api\deps.py
# File chứa các dependencies (phụ thuộc) chung của ứng dụng

# Import AsyncSession từ SQLAlchemy để làm việc với database async
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from fastapi import Depends, HTTPException, status, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import AsyncGenerator, Optional
import os
import httpx
import logging
from dotenv import load_dotenv

load_dotenv()

# Tạo logger
logger = logging.getLogger(__name__)

# Create async engine and session maker
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    # Convert to async URL if needed
    if not DATABASE_URL.startswith("postgresql+asyncpg://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    
    async_engine = create_async_engine(DATABASE_URL)
    async_session_maker = async_sessionmaker(async_engine, expire_on_commit=False)
from services.multifusion_commitanalyst_service import MultifusionCommitAnalystService, MultiFusionV2Service
from services.area_analysis_service import AreaAnalysisService
from services.risk_analysis_service import RiskAnalysisService
from services.han_ai_service import HANAIService
from services.task_service import TaskService
from interfaces.service_factory import service_factory, get_area_analysis_service, get_risk_analysis_service, get_task_service
from interfaces import IAreaAnalysisService, IRiskAnalysisService, ITaskService

# Import OAuth authentication system
from core.security import get_current_user, get_current_user_optional, CurrentUser

# Initialize AI services as singletons
multifusion_commitanalyst_service_instance = MultifusionCommitAnalystService
area_analysis_service_instance = AreaAnalysisService()
risk_analysis_service_instance = RiskAnalysisService()
han_ai_service_instance = HANAIService()
multifusion_v2_service_instance = MultiFusionV2Service() # Add this line
task_service_instance = TaskService()  # Add task service instance

# Register services with factory
service_factory.register_service(IAreaAnalysisService, area_analysis_service_instance)
service_factory.register_service(IRiskAnalysisService, risk_analysis_service_instance)
service_factory.register_service(ITaskService, task_service_instance)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency để lấy database session
    
    Yields:
        AsyncSession: Database session
    """
    if not DATABASE_URL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database configuration not found"
        )
    
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()

def get_multifusion_commitanalyst_service() -> MultifusionCommitAnalystService:
    return multifusion_commitanalyst_service_instance

# Use service factory for interface-based dependencies
# def get_area_analysis_service() -> AreaAnalysisService:
#     return area_analysis_service_instance

# def get_risk_analysis_service() -> RiskAnalysisService:
#     return risk_analysis_service_instance

def get_han_ai_service() -> HANAIService:
    return han_ai_service_instance

# Add this function
def get_multifusion_v2_service() -> MultiFusionV2Service:
    return multifusion_v2_service_instance

# Use service factory for interface-based dependencies
# def get_task_service() -> TaskService:
#     """Dependency để lấy TaskService instance"""
#     return task_service_instance

async def get_current_user_dict(
    authorization: Optional[str] = Header(None)
) -> dict:
    """
    Simple authentication dependency tương tự Dashboard
    Trả về user dict cho backward compatibility với cơ chế localStorage
    Hỗ trợ cả Bearer và token format
    
    Args:
        authorization: Authorization header từ request
        
    Returns:
        dict: Thông tin user hiện tại
        
    Raises:
        HTTPException: Nếu token không hợp lệ hoặc user không tồn tại
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Support both "Bearer " and "token " formats
        if authorization.startswith("Bearer "):
            token = authorization[7:]  # Remove "Bearer " prefix
        elif authorization.startswith("token "):
            token = authorization[6:]  # Remove "token " prefix
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format. Expected 'Bearer <token>' or 'token <token>'",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify token với GitHub API (always use token format for GitHub)
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json"
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            github_user = response.json()
            
            # Return user dict tương tự như Dashboard localStorage format
            return {
                "id": github_user.get("id"),
                "github_id": github_user.get("id"),
                "github_username": github_user.get("login"),
                "username": github_user.get("login"),  # Alias for backward compatibility
                "email": github_user.get("email"),
                "display_name": github_user.get("name"),
                "full_name": github_user.get("name"),
                "avatar_url": github_user.get("avatar_url"),
                "is_active": True,
                "is_verified": True
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )