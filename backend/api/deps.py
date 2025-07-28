# KLTN04\backend\api\deps.py
# File chứa các dependencies (phụ thuộc) chung của ứng dụng

# Import AsyncSession từ SQLAlchemy để làm việc với database async
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import database
from services.multifusion_commitanalyst_service import MultifusionCommitAnalystService
from services.area_analysis_service import AreaAnalysisService
from services.risk_analysis_service import RiskAnalysisService
from services.han_ai_service import HANAIService

# Initialize AI services as singletons
multifusion_commitanalyst_service_instance = MultifusionCommitAnalystService
area_analysis_service_instance = AreaAnalysisService()
risk_analysis_service_instance = RiskAnalysisService()
han_ai_service_instance = HANAIService()

async def get_db() -> AsyncSession:
    async with database.session() as session:
        yield session

def get_multifusion_commitanalyst_service() -> MultifusionCommitAnalystService:
    return multifusion_commitanalyst_service_instance

def get_area_analysis_service() -> AreaAnalysisService:
    return area_analysis_service_instance

def get_risk_analysis_service() -> RiskAnalysisService:
    return risk_analysis_service_instance

def get_han_ai_service() -> HANAIService:
    return han_ai_service_instance