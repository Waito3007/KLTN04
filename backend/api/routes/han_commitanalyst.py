from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from db.database import get_db
from services.han_commitanalyst_service import HanCommitAnalystService
from api.deps import get_han_ai_service
from services.han_ai_service import HANAIService

router = APIRouter(tags=["han-commit-analysis"])

@router.get("/{repo_id}/members/{member_login}/commits-han")
async def get_member_commits_han_analysis(
    repo_id: int,
    member_login: str,
    branch_name: str = None,  # Optional branch filter
    limit: int = 50,
    db: Session = Depends(get_db),
    han_ai_service: HANAIService = Depends(get_han_ai_service)
):
    """Lấy commits của member với HAN AI analysis"""
    try:
        service = HanCommitAnalystService(db, han_ai_service)
        analysis = await service.get_member_commits_with_ai_analysis(
            repo_id, member_login, limit, branch_name
        )
        return {
            "success": True,
            "data": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing member commits with HAN: {str(e)}")

@router.get("/{repo_id}/model-status", response_model=Dict[str, Any])
async def get_ai_model_status(repo_id: int):
    """Kiểm tra trạng thái AI model (HAN Commit Analyzer)"""
    import os
    from pathlib import Path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    han_model_path = Path(current_dir).parent.parent / "ai" / "models" / "han_github_model" / "best_model.pth"
    try:
        model_loaded = han_model_path.exists()
        model_info = {
            "type": "Hierarchical Attention Network (HAN)",
            "purpose": "Commit message analysis and classification",
            "features": [
                "Semantic understanding",
                "Commit type classification",
                "Technology area detection",
                "Multi-task classification"
            ],
            "model_path": str(han_model_path),
            "is_available": model_loaded
        }
        return {
            "success": True,
            "repository_id": repo_id,
            "model_loaded": model_loaded,
            "model_info": model_info
        }
    except Exception as e:
        return {
            "success": False,
            "repository_id": repo_id,
            "model_loaded": False,
            "error": str(e),
            "model_info": {
                "is_available": False,
                "error_details": str(e)
            }
        }
