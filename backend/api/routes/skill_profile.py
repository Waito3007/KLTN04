from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import logging

from db.database import get_db
from services.skill_profile_service import SkillProfileService
from services.area_analysis_service import AreaAnalysisService
from services.risk_analysis_service import RiskAnalysisService
from services.multifusion_commitanalyst_service import MultiFusionV2Service
from api.deps import get_area_analysis_service, get_risk_analysis_service, get_multifusion_v2_service

logger = logging.getLogger(__name__)

skill_profile_router = APIRouter(
    prefix="/api/skill-profile", 
    tags=["Skill Profile"]
)

def get_skill_profile_service(
    db: Session = Depends(get_db),
    area_service: AreaAnalysisService = Depends(get_area_analysis_service),
    risk_service: RiskAnalysisService = Depends(get_risk_analysis_service),
    multifusion_service: MultiFusionV2Service = Depends(get_multifusion_v2_service)
) -> SkillProfileService:
    return SkillProfileService(
        db=db, 
        area_analysis_service=area_service, 
        risk_analysis_service=risk_service, 
        multifusion_service=multifusion_service
    )

@skill_profile_router.get("/repository/{repo_id}/member/{member_login}")
async def get_skill_profile(
    repo_id: int,
    member_login: str,
    skill_profile_service: SkillProfileService = Depends(get_skill_profile_service)
):
    """
    Tạo và trả về hồ sơ năng lực toàn diện cho một thành viên trong repository.
    
    Tổng hợp dữ liệu từ các mô hình AI:
    - Commit Analyst (Loại commit)
    - Risk Analyst (Mức độ rủi ro)
    - Area Analyst (Lĩnh vực phát triển)
    """
    try:
        logger.info(f"Generating skill profile for member '{member_login}' in repo {repo_id}")
        profile = skill_profile_service.generate_skill_profile(repo_id, member_login)
        return profile
    except Exception as e:
        logger.error(f"Error generating skill profile for {member_login}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred while generating the skill profile: {str(e)}"
        )