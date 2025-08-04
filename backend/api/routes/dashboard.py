"""
Dashboard API Routes - Endpoints cho analytics dashboard
Tuân thủ quy tắc KLTN04: Validation, error handling, structured logging
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from api.deps import get_current_user, get_db
from core.security import CurrentUser  # Import CurrentUser from core.security
from services.dashboard_analytics_service import DashboardAnalyticsService
from schemas.dashboard import (
    DashboardAnalyticsResponse, 
    ProgressAnalyticsResponse,
    RiskAnalyticsResponse, 
    AssignmentSuggestionsResponse,
    ProductivityMetricsResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get(
    "/analytics/{repo_owner}/{repo_name}", 
    response_model=DashboardAnalyticsResponse,
    summary="Lấy phân tích toàn diện cho dashboard",
    description="Trả về tất cả analytics bao gồm tiến độ, rủi ro, và gợi ý phân công"
)
async def get_comprehensive_analytics(
    repo_owner: str,
    repo_name: str,
    days_back: int = Query(default=30, ge=1, le=365, description="Số ngày phân tích (1-365)"),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> DashboardAnalyticsResponse:
    """
    Lấy phân tích toàn diện cho dashboard
    
    Args:
        repo_owner: Chủ sở hữu repository
        repo_name: Tên repository
        days_back: Số ngày để phân tích (mặc định 30)
        current_user: User hiện tại (từ JWT token)
        db: Database session
    
    Returns:
        DashboardAnalyticsResponse: Dữ liệu analytics đầy đủ
    
    Raises:
        HTTPException: 404 nếu repo không tồn tại, 403 nếu không có quyền truy cập
    """
    try:
        # logger.info(f"Getting comprehensive analytics for {repo_owner}/{repo_name} by user {current_user.github_username}")
        
        # # Validate repository access
        # await _validate_repo_access(repo_owner, repo_name, current_user, db)
        
        # Initialize service
        analytics_service = DashboardAnalyticsService(db)
        
        # Get analytics
        analytics_data = await analytics_service.get_comprehensive_analytics(
            repo_owner=repo_owner,
            repo_name=repo_name,
            days_back=days_back
        )
        
        logger.info(f"Successfully generated analytics for {repo_owner}/{repo_name}")
        return DashboardAnalyticsResponse(**analytics_data)
        
    except ValueError as e:
        logger.warning(f"Invalid request for analytics: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get comprehensive analytics: {e}")
        raise HTTPException(status_code=500, detail="Lỗi internal khi lấy analytics")

@router.get(
    "/progress/{repo_owner}/{repo_name}",
    response_model=ProgressAnalyticsResponse,
    summary="Lấy phân tích tiến độ dự án",
    description="Phân tích tiến độ, velocity, và productivity của dự án"
)
async def get_progress_analytics(
    repo_owner: str,
    repo_name: str,
    days_back: int = Query(default=30, ge=1, le=365),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ProgressAnalyticsResponse:
    """Lấy chi tiết phân tích tiến độ"""
    try:
        logger.info(f"Getting progress analytics for {repo_owner}/{repo_name}")
        
        await _validate_repo_access(repo_owner, repo_name, current_user, db)
        
        analytics_service = DashboardAnalyticsService(db)
        
        # Get repo ID
        repo_id = await analytics_service._get_repo_id(repo_owner, repo_name)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository không tồn tại")
        
        # Get progress analysis
        progress_analysis = await analytics_service._analyze_progress(repo_id, days_back)
        
        return ProgressAnalyticsResponse(
            repository={"owner": repo_owner, "name": repo_name, "id": repo_id},
            analysis_period={
                "days": days_back,
                "start_date": f"{days_back} ngày trước",
                "end_date": "hôm nay"
            },
            progress=progress_analysis.__dict__
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get progress analytics: {e}")
        raise HTTPException(status_code=500, detail="Lỗi internal khi phân tích tiến độ")

@router.get(
    "/risks/{repo_owner}/{repo_name}",
    response_model=RiskAnalyticsResponse,
    summary="Lấy phân tích rủi ro dự án",
    description="Phân tích rủi ro, cảnh báo, và gợi ý giảm thiểu"
)
async def get_risk_analytics(
    repo_owner: str,
    repo_name: str,
    days_back: int = Query(default=30, ge=1, le=365),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> RiskAnalyticsResponse:
    """Lấy chi tiết phân tích rủi ro"""
    try:
        logger.info(f"Getting risk analytics for {repo_owner}/{repo_name}")
        
        await _validate_repo_access(repo_owner, repo_name, current_user, db)
        
        analytics_service = DashboardAnalyticsService(db)
        
        repo_id = await analytics_service._get_repo_id(repo_owner, repo_name)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository không tồn tại")
        
        risk_analysis = await analytics_service._analyze_risks(repo_id, days_back)
        
        return RiskAnalyticsResponse(
            repository={"owner": repo_owner, "name": repo_name, "id": repo_id},
            analysis_period={
                "days": days_back,
                "start_date": f"{days_back} ngày trước", 
                "end_date": "hôm nay"
            },
            risks=risk_analysis.__dict__
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get risk analytics: {e}")
        raise HTTPException(status_code=500, detail="Lỗi internal khi phân tích rủi ro")

@router.get(
    "/assignments/{repo_owner}/{repo_name}",
    response_model=AssignmentSuggestionsResponse,
    summary="Lấy gợi ý phân công công việc",
    description="AI-powered gợi ý phân công dựa trên skills và workload"
)
async def get_assignment_suggestions(
    repo_owner: str,
    repo_name: str,
    days_back: int = Query(default=30, ge=1, le=365),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> AssignmentSuggestionsResponse:
    """Lấy gợi ý phân công thông minh"""
    try:
        logger.info(f"Getting assignment suggestions for {repo_owner}/{repo_name}")
        
        await _validate_repo_access(repo_owner, repo_name, current_user, db)
        
        analytics_service = DashboardAnalyticsService(db)
        
        repo_id = await analytics_service._get_repo_id(repo_owner, repo_name)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository không tồn tại")
        
        suggestions = await analytics_service._generate_assignment_suggestions(repo_id, days_back)
        
        return AssignmentSuggestionsResponse(
            repository={"owner": repo_owner, "name": repo_name, "id": repo_id},
            analysis_period={
                "days": days_back,
                "start_date": f"{days_back} ngày trước",
                "end_date": "hôm nay"
            },
            suggestions=[s.__dict__ for s in suggestions]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get assignment suggestions: {e}")
        raise HTTPException(status_code=500, detail="Lỗi internal khi tạo gợi ý phân công")

@router.get(
    "/productivity/{repo_owner}/{repo_name}",
    response_model=ProductivityMetricsResponse,
    summary="Lấy metrics productivity team",
    description="Thống kê chi tiết về productivity và performance team"
)
async def get_productivity_metrics(
    repo_owner: str,
    repo_name: str,
    days_back: int = Query(default=30, ge=1, le=365),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ProductivityMetricsResponse:
    """Lấy metrics productivity chi tiết"""
    try:
        logger.info(f"Getting productivity metrics for {repo_owner}/{repo_name}")
        
        await _validate_repo_access(repo_owner, repo_name, current_user, db)
        
        analytics_service = DashboardAnalyticsService(db)
        
        repo_id = await analytics_service._get_repo_id(repo_owner, repo_name)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository không tồn tại")
        
        metrics = await analytics_service._get_team_productivity_metrics(repo_id, days_back)
        
        return ProductivityMetricsResponse(
            repository={"owner": repo_owner, "name": repo_name, "id": repo_id},
            analysis_period={
                "days": days_back,
                "start_date": f"{days_back} ngày trước",
                "end_date": "hôm nay"
            },
            metrics=metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get productivity metrics: {e}")
        raise HTTPException(status_code=500, detail="Lỗi internal khi lấy productivity metrics")

@router.get(
    "/insights/{repo_owner}/{repo_name}",
    summary="Lấy insights nhanh cho dashboard",
    description="Các insights và highlights quan trọng cho dashboard"
)
async def get_dashboard_insights(
    repo_owner: str,
    repo_name: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Lấy insights nhanh cho dashboard header"""
    try:
        logger.info(f"Getting dashboard insights for {repo_owner}/{repo_name}")
        
        await _validate_repo_access(repo_owner, repo_name, current_user, db)
        
        analytics_service = DashboardAnalyticsService(db)
        
        repo_id = await analytics_service._get_repo_id(repo_owner, repo_name)
        if not repo_id:
            raise HTTPException(status_code=404, detail="Repository không tồn tại")
        
        # Get quick insights (last 7 days)
        recent_commits = await analytics_service._get_commits_with_analysis(repo_id, 7)
        
        # Calculate quick metrics
        total_commits_week = len(recent_commits)
        high_risk_commits = len([c for c in recent_commits if c.get('risk_level') == 'highrisk'])
        fix_commits = len([c for c in recent_commits if c.get('commit_type') == 'fix'])
        
        # Get current tasks count
        current_tasks = await analytics_service._get_current_tasks(repo_id)
        todo_tasks = len([t for t in current_tasks if t.get('status') == 'TODO'])
        in_progress_tasks = len([t for t in current_tasks if t.get('status') == 'IN_PROGRESS'])
        
        return {
            "quick_stats": {
                "commits_this_week": total_commits_week,
                "high_risk_commits": high_risk_commits,
                "fix_ratio": round((fix_commits / total_commits_week * 100) if total_commits_week > 0 else 0, 1),
                "pending_tasks": todo_tasks,
                "active_tasks": in_progress_tasks
            },
            "health_indicators": {
                "code_health": "good" if high_risk_commits < total_commits_week * 0.2 else "warning",
                "velocity": "normal" if total_commits_week >= 5 else "slow",
                "task_progress": "on_track" if in_progress_tasks > 0 else "stalled"
            },
            "alerts": _generate_quick_alerts(total_commits_week, high_risk_commits, todo_tasks, in_progress_tasks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dashboard insights: {e}")
        raise HTTPException(status_code=500, detail="Lỗi internal khi lấy insights")

# Helper functions
async def _validate_repo_access(
    repo_owner: str, 
    repo_name: str, 
    current_user: CurrentUser, 
    db: AsyncSession
) -> None:
    """Validate user có quyền truy cập repository"""
    # Simplified access check - in production should check collaborators table
    # For now, allow access if user exists
    if not current_user.github_username:
        raise HTTPException(status_code=403, detail="Cần GitHub username để truy cập")

def _generate_quick_alerts(
    commits_week: int, 
    high_risk_commits: int, 
    todo_tasks: int, 
    in_progress_tasks: int
) -> List[Dict[str, str]]:
    """Tạo alerts nhanh cho dashboard"""
    alerts = []
    
    if high_risk_commits > commits_week * 0.3:
        alerts.append({
            "type": "warning",
            "message": f"Có {high_risk_commits} commits rủi ro cao trong tuần"
        })
    
    if commits_week < 3:
        alerts.append({
            "type": "info", 
            "message": "Velocity thấp - chỉ có {commits_week} commits tuần này"
        })
    
    if todo_tasks > 20:
        alerts.append({
            "type": "warning",
            "message": f"Có {todo_tasks} tasks đang chờ xử lý"
        })
    
    if in_progress_tasks == 0 and todo_tasks > 0:
        alerts.append({
            "type": "info",
            "message": "Không có task nào đang được thực hiện"
        })
    
    return alerts
