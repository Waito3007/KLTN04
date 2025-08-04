# KLTN04\backend\api\deps.py
# File chứa các dependencies (phụ thuộc) chung của ứng dụng

# Import AsyncSession từ SQLAlchemy để làm việc với database async
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import AsyncGenerator, Optional
import os
from dotenv import load_dotenv

load_dotenv()

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

# Import OAuth authentication system
from core.security import get_current_user, get_current_user_optional, CurrentUser

# Initialize AI services as singletons
multifusion_commitanalyst_service_instance = MultifusionCommitAnalystService
area_analysis_service_instance = AreaAnalysisService()
risk_analysis_service_instance = RiskAnalysisService()
han_ai_service_instance = HANAIService()
multifusion_v2_service_instance = MultiFusionV2Service() # Add this line
task_service_instance = TaskService()  # Add task service instance

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

def get_area_analysis_service() -> AreaAnalysisService:
    return area_analysis_service_instance

def get_risk_analysis_service() -> RiskAnalysisService:
    return risk_analysis_service_instance

def get_han_ai_service() -> HANAIService:
    return han_ai_service_instance

# Add this function
def get_multifusion_v2_service() -> MultiFusionV2Service:
    return multifusion_v2_service_instance

def get_task_service() -> TaskService:
    """Dependency để lấy TaskService instance"""
    return task_service_instance

async def get_current_user_dict(
    current_user: CurrentUser = Depends(get_current_user)
) -> dict:
    """
    Dependency wrapper để trả về user dict cho backward compatibility
    Sử dụng OAuth authentication system từ core.security
    
    Args:
        current_user: CurrentUser object từ OAuth system
        
    Returns:
        dict: Thông tin user hiện tại
        
    Raises:
        HTTPException: Nếu token không hợp lệ hoặc user không tồn tại
    """
    return current_user.to_dict()