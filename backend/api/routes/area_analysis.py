from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from services.area_analysis_service import AreaAnalysisService
from schemas.commit import CommitAreaAnalysisRequest
import logging

logger = logging.getLogger(__name__)

area_analysis_router = APIRouter(prefix="/api/area-analysis", tags=["Area Analysis"])

# Initialize AreaAnalysisService
area_analysis_service = AreaAnalysisService()

@area_analysis_router.post("/predict")
async def predict_commit_area(commit_data: CommitAreaAnalysisRequest) -> Dict[str, str]:
    """
    Phân tích phạm vi công việc (dev area) của một commit dựa trên message và diff.
    
    Args:
        commit_data (CommitAreaAnalysisRequest): Dữ liệu commit bao gồm commit_message, diff_content,
                                                files_count, lines_added, lines_removed, total_changes.
    
    Returns:
        Dict[str, str]: Kết quả phân tích với dev_area được dự đoán.
    """
    try:
        predicted_area = area_analysis_service.predict_area(commit_data.dict())
        return {"dev_area": predicted_area}
    except Exception as e:
        logger.error(f"Error analyzing commit area: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze commit area: {str(e)}")
