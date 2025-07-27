from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
from db.database import get_db
from services.assignment_recommendation_service import AssignmentRecommendationService
from services.multifusion_commitanalyst_service import MultifusionCommitAnalystService
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
        multifusion_commit_analyst = MultifusionCommitAnalystService(None)
        area_service = AreaAnalysisService()
        risk_service = RiskAnalysisService()
        
        return {
            "success": True,
            "models": {
                "multifusion_commit_analyst": {
                    "name": "MultiFusion Commit Analyst",
                    "purpose": "Commit Type Classification",
                    "status": "available" if multifusion_commit_analyst.is_model_available() else "unavailable",
                    "features": [
                        "BERT-based semantic analysis",
                        "Code metrics integration", 
                        "Programming language detection",
                        "Multi-modal fusion"
                    ],
                    "supported_commit_types": list(multifusion_commit_analyst.label_map.keys()) if multifusion_commit_analyst.is_model_available() and multifusion_commit_analyst.label_map else []
                },
                "area_analyst": {
                    "name": "Area Analysis Model",
                    "purpose": "Development Area Classification (Frontend/Backend/Database...)",
                    "status": "available" if area_service.model is not None else "unavailable",
                    "features": [
                        "BERT + MLP fusion",
                        "Code area classification",
                        "File-based analysis"
                    ],
                    "supported_areas": list(area_service.label_encoder.classes_) if area_service.model is not None else []
                },
                "risk_analyst": {
                    "name": "Risk Analysis Model", 
                    "purpose": "Commit Risk Level Assessment (High/Low Risk)",
                    "status": "available" if risk_service.model is not None else "unavailable",
                    "features": [
                        "Risk assessment",
                        "Change impact analysis",
                        "BERT + numerical features"
                    ],
                    "supported_risk_levels": ["lowrisk", "highrisk"]
                }
            },
            "integration": {
                "assignment_recommendation": True,
                "fallback_to_legacy": True,
                "ai_coverage_threshold": 0.5,
                "enhanced_scoring": True
            },
            "performance": {
                "ai_enhanced_accuracy": "Improved commit type and area detection",
                "recommendation_quality": "Higher precision with AI models",
                "fallback_available": "Legacy analysis as backup"
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking AI models status: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking AI models: {str(e)}")

@router.get("/test-assignment-recommendation/{repo_id}")
async def test_ai_assignment_recommendation(
    repo_id: int,
    task_description: str = "Implement new API endpoint for user management",
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Test endpoint để kiểm tra Assignment Recommendation với AI models
    """
    try:
        service = AssignmentRecommendationService(db)
        
        # Analyze task to determine characteristics
        task_type = "feature"
        task_area = "backend" 
        risk_level = "medium"
        
        # Simple heuristics
        description_lower = task_description.lower()
        if any(word in description_lower for word in ["fix", "bug", "error"]):
            task_type = "fix"
        elif any(word in description_lower for word in ["frontend", "ui", "react"]):
            task_area = "frontend"
        elif any(word in description_lower for word in ["database", "sql"]):
            task_area = "database"
            
        # Get AI-enhanced recommendations
        recommendations = service.recommend_assignees(
            repository_id=repo_id,
            task_type=task_type,
            task_area=task_area,
            risk_level=risk_level,
            required_skills=["python"],
            top_k=3
        )
        
        # Analyze member skills to see AI coverage
        member_skills = service.analyze_member_skills(repo_id)
        
        ai_stats = {
            "total_members": len(member_skills),
            "ai_enhanced_members": sum(1 for profile in member_skills.values() if profile.get('ai_coverage', 0) > 0.5),
            "avg_ai_coverage": sum(profile.get('ai_coverage', 0) for profile in member_skills.values()) / len(member_skills) if member_skills else 0
        }
        
        return {
            "success": True,
            "test_input": {
                "repository_id": repo_id,
                "task_description": task_description,
                "detected_task_type": task_type,
                "detected_area": task_area,
                "risk_level": risk_level
            },
            "recommendations": recommendations,
            "ai_analysis_stats": ai_stats,
            "model_usage": {
                "multifusion_v2": "Used for commit type classification",
                "area_analyst": "Used for development area detection", 
                "risk_analyst": "Used for risk level assessment"
            }
        }
        
    except Exception as e:
        logger.error(f"Error testing AI assignment recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Error testing: {str(e)}")

@router.post("/compare-analysis")
async def compare_legacy_vs_ai_analysis(
    repo_id: int,
    member_login: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    So sánh kết quả phân tích giữa legacy methods và AI models cho một member
    """
    try:
        service = AssignmentRecommendationService(db)
        
        # Get analysis results (will include both legacy and AI)
        member_skills = service.analyze_member_skills(repo_id)
        
        if member_login not in member_skills:
            raise HTTPException(status_code=404, detail=f"Member {member_login} not found")
        
        profile = member_skills[member_login]
        
        return {
            "success": True,
            "member": member_login,
            "analysis_comparison": {
                "ai_coverage": profile.get('ai_coverage', 0),
                "total_commits": profile['total_commits'],
                "ai_analyzed_commits": profile.get('ai_analysis_count', 0),
                "legacy_analysis": {
                    "commit_types": dict(profile['commit_types']),
                    "areas": dict(profile['areas']),
                    "risk_levels": dict(profile['risk_levels']),
                    "risk_tolerance": profile['risk_tolerance']
                },
                "ai_predictions": profile.get('ai_predictions', {}),
                "expertise_areas": profile['expertise_areas'],
                "recommendation": "AI-enhanced" if profile.get('ai_coverage', 0) > 0.5 else "Legacy-based"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error comparing: {str(e)}")
