from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional


import logging
from db.database import get_db
from services.multifusion_commitanalyst_service import MultifusionCommitAnalystService, MultiFusionV2Service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["multifusion-commit-analysis"])

@router.get("/{repo_id}/ai/model-v2-status")
async def get_multifusion_v2_status(repo_id: int):
    """Kiểm tra trạng thái và thông tin của MultiFusion V2 model."""
    try:
        multifusion_v2 = MultiFusionV2Service()
        if not hasattr(multifusion_v2, "get_model_info"):
            logger.error("MultiFusionV2Service missing get_model_info method")
            raise HTTPException(status_code=500, detail="MultiFusionV2Service missing get_model_info method")
        model_info = multifusion_v2.get_model_info()
        return {"success": True, "repository_id": repo_id, "model_info": model_info}
    except Exception as e:
        try:
            logger.error(f"Error in get_multifusion_v2_status: {e}", exc_info=True)
        except Exception:
            print(f"Error in get_multifusion_v2_status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{repo_id}/commits/all/analysis")
async def get_all_repo_commits_analysis(
    repo_id: int,
    limit: int = 100,
    offset: int = 0,
    branch_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Lấy tất cả commits của repo với AI analysis."""
    try:
        service = MultifusionCommitAnalystService(db)
        analysis = await service.get_all_repo_commits_with_analysis(
            repo_id, limit, offset, branch_name
        )
        return {
            "success": True,
            "repository_id": repo_id,
            **analysis
        }
    except Exception as e:
        logger.error(f"Error getting all repo commits analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting all repo commits analysis: {str(e)}")