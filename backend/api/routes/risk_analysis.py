from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from services.risk_analysis_service import RiskAnalysisService
from schemas.commit import CommitAreaAnalysisRequest # Reusing schema for now, might need a dedicated one
import logging

logger = logging.getLogger(__name__)

risk_analysis_router = APIRouter(prefix="/api/risk-analysis", tags=["Risk Analysis"])

# Initialize RiskAnalysisService
risk_analysis_service = RiskAnalysisService()

@risk_analysis_router.post("/predict")
async def predict_commit_risk(commit_data: CommitAreaAnalysisRequest) -> Dict[str, str]:
    """
    Phân tích độ rủi ro của một commit dựa trên message và diff.
    
    Args:
        commit_data (CommitAreaAnalysisRequest): Dữ liệu commit bao gồm commit_message, diff_content,
                                                files_count, lines_added, lines_removed, total_changes.
    
    Returns:
        Dict[str, str]: Kết quả phân tích với risk được dự đoán (lowrisk/highrisk).
    """
    try:
        predicted_risk = risk_analysis_service.predict_risk(commit_data.dict())
        return {"risk": predicted_risk}
    except Exception as e:
        logger.error(f"Error analyzing commit risk: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze commit risk: {str(e)}")
