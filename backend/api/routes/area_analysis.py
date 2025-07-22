from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from services.area_analysis_service import AreaAnalysisService
from schemas.commit import CommitAreaAnalysisRequest
import logging
from fastapi import Depends
from sqlalchemy.orm import Session
from db.database import get_db
from services.member_analysis_service import MemberAnalysisService

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

@area_analysis_router.get("/repositories/{repo_id}/full-area-analysis")
async def get_full_area_analysis(
    repo_id: int,
    limit_per_member: int = 50, # Default limit for commits per member
    db: Session = Depends(get_db)
):
    """
    Thực hiện phân tích khu vực phát triển toàn diện cho tất cả thành viên trong một repository.
    
    Args:
        repo_id: ID của repository.
        limit_per_member: Số lượng commit tối đa để phân tích cho mỗi thành viên.
    
    Returns:
        Dict[str, Any]: Kết quả phân tích khu vực phát triển tổng hợp cho tất cả thành viên.
    """
    try:
        member_service = MemberAnalysisService(db)
        members = member_service.get_repository_members(repo_id)
        
        full_area_results = {
            "success": True,
            "repository_id": repo_id,
            "total_members": len(members),
            "total_commits_analyzed": 0,
            "area_distribution": {},
            "members_area_analysis": []
        }
        
        if not area_analysis_service.model:
            raise HTTPException(status_code=503, detail="Area analysis model not loaded or available.")
        
        for member in members:
            member_login = member['login']
            member_commits_data = member_service._get_member_commits_raw(repo_id, member_login, limit_per_member)
            
            member_area_summary = {"total_commits": len(member_commits_data)}
            
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
                    predicted_area = area_analysis_service.predict_area(commit_data_for_analysis)
                    member_area_summary[predicted_area] = member_area_summary.get(predicted_area, 0) + 1
                    full_area_results["area_distribution"][predicted_area] = full_area_results["area_distribution"].get(predicted_area, 0) + 1
                except Exception as e:
                    member_area_summary["unknown"] = member_area_summary.get("unknown", 0) + 1
                    full_area_results["area_distribution"]["unknown"] = full_area_results["area_distribution"].get("unknown", 0) + 1
                    logger.error(f"Error predicting area for commit {commit.sha} by {member_login}: {e}")
                full_area_results["total_commits_analyzed"] += 1
            
            full_area_results["members_area_analysis"].append({
                "member_login": member_login,
                "area_summary": member_area_summary
            })
                
        return full_area_results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting full area analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting full area analysis: {str(e)}")
