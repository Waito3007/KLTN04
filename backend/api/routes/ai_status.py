from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
from db.database import get_db
from services.multifusion_commitanalyst_service import MultifusionCommitAnalystService, MultiFusionV2Service
from services.area_analysis_service import AreaAnalysisService
from services.risk_analysis_service import RiskAnalysisService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai-status", tags=["AI Status"])

@router.get("/assignment-models")
async def get_assignment_ai_models_status() -> Dict[str, Any]:
    """
    Kiểm tra trạng thái của các AI models được sử dụng trong Assignment Recommendation
    """
    try:
        # Initialize services
        multifusion_v2_service = MultiFusionV2Service()
        area_service = AreaAnalysisService()
        risk_service = RiskAnalysisService()
        
        return {
            "success": True,
            "models": {
                "multifusion_commit_analyst": {
                    "name": "MultiFusion Commit Analyst",
                    "purpose": "Commit Type Classification",
                    "status": "available" if multifusion_v2_service.is_model_available() else "unavailable",
                    "features": [
                        "CodeBERT-based semantic analysis",
                        "Numerical feature integration", 
                        "Multi-modal fusion"
                    ],
                    "supported_commit_types": list(multifusion_v2_service.label_map.keys()) if multifusion_v2_service.is_model_available() and multifusion_v2_service.label_map else []
                },
                "area_analyst": {
                    "name": "Area Analysis Model",
                    "purpose": "Development Area Classification (Frontend/Backend/Database...)",
                    "status": "available" if area_service.model is not None else "unavailable",
                    "features": [
                        "DistilBERT + MLP fusion",
                        "Code area classification"
                    ],
                    "supported_areas": list(area_service.label_encoder.classes_) if area_service.model is not None else []
                },
                "risk_analyst": {
                    "name": "Risk Analysis Model", 
                    "purpose": "Commit Risk Level Assessment (High/Low Risk)",
                    "status": "available" if risk_service.model is not None else "unavailable",
                    "features": [
                        "DistilBERT + numerical features",
                        "Risk assessment"
                    ],
                    "supported_risk_levels": ["lowrisk", "highrisk"]
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking AI models status: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking AI models: {str(e)}")
