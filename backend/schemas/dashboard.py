"""
Pydantic Schemas for Dashboard Analytics API Responses
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# =================================================================
# Common Schemas (Các schema dùng chung)
# =================================================================

class RepositoryInfo(BaseModel):
    """Thông tin cơ bản của repository."""
    owner: str
    name: str
    id: int

class AnalysisPeriod(BaseModel):
    """Khung thời gian phân tích."""
    days: int
    start_date: str
    end_date: str

# =================================================================
# Progress Analytics Schemas (Schema cho phân tích tiến độ)
# =================================================================

class CommitTrendItem(BaseModel):
    """Dữ liệu cho biểu đồ xu hướng commit."""
    date: str
    commits: int

class ProgressAnalytics(BaseModel):
    """Chi tiết kết quả phân tích tiến độ."""
    total_commits: int
    commits_by_type: Dict[str, int]
    commits_by_area: Dict[str, int]
    commits_trend: List[CommitTrendItem]
    velocity: float
    productivity_score: float
    recommendations: List[str]

class ProgressAnalyticsResponse(BaseModel):
    """Schema cho API trả về phân tích tiến độ."""
    repository: RepositoryInfo
    analysis_period: AnalysisPeriod
    progress: ProgressAnalytics

# =================================================================
# Risk Analytics Schemas (Schema cho phân tích rủi ro)
# =================================================================

class RiskTrendItem(BaseModel):
    """Dữ liệu cho biểu đồ xu hướng rủi ro."""
    date: str
    risk_percentage: float
    high_risk_commits: int
    total_commits: int

class RiskAnalytics(BaseModel):
    """Chi tiết kết quả phân tích rủi ro."""
    high_risk_commits: List[Dict[str, Any]] = Field(description="Top 10 commits có rủi ro cao nhất")
    risk_trend: List[RiskTrendItem]
    risk_score: float
    critical_areas: List[str]
    warnings: List[str]
    mitigation_suggestions: List[str]

class RiskAnalyticsResponse(BaseModel):
    """Schema cho API trả về phân tích rủi ro."""
    repository: RepositoryInfo
    analysis_period: AnalysisPeriod
    risks: RiskAnalytics

# =================================================================
# Assignment Suggestion Schemas (Schema cho gợi ý phân công)
# =================================================================

class SuggestedTask(BaseModel):
    """Thông tin về một task được gợi ý."""
    task_id: Optional[int]
    title: Optional[str]
    priority: Optional[str]
    match_score: float
    estimated_effort: str

class AssignmentSuggestion(BaseModel):
    """Chi tiết một gợi ý phân công cho thành viên."""
    member_id: str
    member_name: str
    expertise_areas: List[str]
    suggested_tasks: List[SuggestedTask]
    workload_score: float
    availability_score: float
    skill_match_score: float
    rationale: str

class AssignmentSuggestionsResponse(BaseModel):
    """Schema cho API trả về các gợi ý phân công."""
    repository: RepositoryInfo
    analysis_period: AnalysisPeriod
    suggestions: List[AssignmentSuggestion]

# =================================================================
# Productivity Metrics Schemas (Schema cho chỉ số năng suất)
# =================================================================

class TeamSummary(BaseModel):
    """Tóm tắt các chỉ số của cả team."""
    total_commits: int
    total_lines_changed: int
    average_commit_size: float
    fix_ratio: float
    active_contributors: int

class MemberMetric(BaseModel):
    """Chi tiết chỉ số của một thành viên."""
    commits: int
    lines_added: int
    lines_removed: int
    files_changed: int
    commit_types: Dict[str, int]
    areas: Dict[str, int]

class DailyCommitTrend(BaseModel):
    """Dữ liệu cho biểu đồ xu hướng commit hàng ngày (chi tiết hơn)."""
    date: str
    commits: int
    lines_changed: int

class ProductivityTrends(BaseModel):
    """Các chỉ số xu hướng về năng suất."""
    daily_commits: List[DailyCommitTrend]
    weekly_velocity: float

class ProductivityMetrics(BaseModel):
    """Toàn bộ các chỉ số về năng suất."""
    team_summary: TeamSummary
    member_metrics: Dict[str, MemberMetric]
    trends: ProductivityTrends

class ProductivityMetricsResponse(BaseModel):
    """Schema cho API trả về các chỉ số năng suất."""
    repository: RepositoryInfo
    analysis_period: AnalysisPeriod
    metrics: ProductivityMetrics

# =================================================================
# Comprehensive Dashboard Analytics Schema (Schema tổng hợp)
# =================================================================

class DashboardAnalyticsResponse(BaseModel):
    """Schema cho API trả về phân tích tổng hợp cho dashboard."""
    repository: RepositoryInfo
    analysis_period: AnalysisPeriod
    progress: ProgressAnalytics
    risks: RiskAnalytics
    assignment_suggestions: List[AssignmentSuggestion]
    productivity_metrics: ProductivityMetrics
    generated_at: datetime
