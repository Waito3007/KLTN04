from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from schemas.commit import CommitAreaAnalysisRequest # Reusing schema for now, might need a dedicated one
import logging
from fastapi import Depends
from sqlalchemy.orm import Session
from db.database import get_db
from services.member_analysis_service import MemberAnalysisService
from interfaces.service_factory import get_risk_analysis_service
from interfaces import IRiskAnalysisService

logger = logging.getLogger(__name__)

risk_analysis_router = APIRouter(prefix="/api/risk-analysis", tags=["Risk Analysis"])

@risk_analysis_router.post("/predict")
async def predict_commit_risk(
    commit_data: CommitAreaAnalysisRequest,
    risk_analysis_service: IRiskAnalysisService = Depends(get_risk_analysis_service)
) -> Dict[str, str]:
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

@risk_analysis_router.get("/repositories/{repo_id}/full-risk-analysis")
async def get_full_risk_analysis(
    repo_id: int,
    limit_per_member: int = 1000, # Increased default limit for commits per member
    branch_name: str = None, # Optional branch filter
    db: Session = Depends(get_db),
    risk_analysis_service: IRiskAnalysisService = Depends(get_risk_analysis_service)
):
    """
    Thực hiện phân tích rủi ro toàn diện cho tất cả thành viên trong một repository.
    
    Args:
        repo_id: ID của repository.
        limit_per_member: Số lượng commit tối đa để phân tích cho mỗi thành viên.
        branch_name: Tên branch để lọc commits (tùy chọn).
    
    Returns:
        Dict[str, Any]: Kết quả phân tích rủi ro tổng hợp cho tất cả thành viên.
    """
    try:
        member_service = MemberAnalysisService(db)
        members = member_service.get_repository_members(repo_id)
        
        full_risk_results = {
            "success": True,
            "repository_id": repo_id,
            "branch_name": branch_name,
            "total_members": len(members),
            "total_commits_analyzed": 0,
            "risk_distribution": {"lowrisk": 0, "highrisk": 0, "unknown": 0},
            "members_risk_analysis": []
        }
        
        if not risk_analysis_service.model:
            raise HTTPException(status_code=503, detail="Risk analysis model not loaded or available.")
        
        for member in members:
            member_login = member['login']
            member_commits_data = member_service._get_member_commits_raw(repo_id, member_login, limit_per_member, branch_name)
            
            member_risk_summary = {"lowrisk": 0, "highrisk": 0, "unknown": 0, "total_commits": len(member_commits_data)}
            
            for commit in member_commits_data:
                commit_data_for_analysis = {
                    "commit_message": commit[2] or '',  # message
                    "diff_content": commit[8] or '',  # diff_content
                    "files_count": commit[6] or 0,  # files_changed
                    "lines_added": commit[4] or 0,  # insertions
                    "lines_removed": commit[5] or 0,  # deletions
                    "total_changes": (commit[4] or 0) + (commit[5] or 0)
                }
                try:
                    predicted_risk = risk_analysis_service.predict_risk(commit_data_for_analysis)
                    member_risk_summary[predicted_risk] = member_risk_summary.get(predicted_risk, 0) + 1
                    full_risk_results["risk_distribution"][predicted_risk] = full_risk_results["risk_distribution"].get(predicted_risk, 0) + 1
                except Exception as e:
                    member_risk_summary["unknown"] += 1
                    full_risk_results["risk_distribution"]["unknown"] += 1
                    logger.error(f"Error predicting risk for commit {commit.sha} by {member_login}: {e}")
                full_risk_results["total_commits_analyzed"] += 1
            
            full_risk_results["members_risk_analysis"].append({
                "member_login": member_login,
                "risk_summary": member_risk_summary
            })
                
        return full_risk_results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting full risk analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting full risk analysis: {str(e)}")
