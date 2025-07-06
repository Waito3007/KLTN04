# KLTN04\backend\api\deps.py
# File chứa các dependencies (phụ thuộc) chung của ứng dụng

# Import AsyncSession từ SQLAlchemy để làm việc với database async
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import database
from services.multifusion_ai_service import MultiFusionAIService
from services.multifusion_v2_service import MultiFusionV2Service

async def get_db() -> AsyncSession:
    async with database.session() as session:
        yield session

def get_multifusion_ai_service() -> MultiFusionAIService:
    return MultiFusionAIService()

def get_multifusion_v2_service() -> MultiFusionV2Service:
    return MultiFusionV2Service()