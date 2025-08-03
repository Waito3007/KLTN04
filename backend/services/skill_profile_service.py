from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Dict, Any, List
import logging

from services.area_analysis_service import AreaAnalysisService
from services.risk_analysis_service import RiskAnalysisService
from services.multifusion_commitanalyst_service import MultifusionCommitAnalystService, MultiFusionV2Service
from collections import defaultdict

logger = logging.getLogger(__name__)

class SkillProfileService:
    def __init__(
        self,
        db: Session,
        area_analysis_service: AreaAnalysisService,
        risk_analysis_service: RiskAnalysisService,
        multifusion_service: MultiFusionV2Service
    ):
        self.db = db
        self.area_analysis_service = area_analysis_service
        self.risk_analysis_service = risk_analysis_service
        self.multifusion_service = multifusion_service
        self.commit_analyst_service = MultifusionCommitAnalystService(db) # Helper service

    def _get_member_commits(self, repo_id: int, member_login: str) -> List[Dict[str, Any]]:
        """Lấy tất cả commits của một thành viên trong repo."""
        query = text("""
            SELECT 
                id, sha, message, diff_content, files_changed, 
                insertions, deletions, modified_files, file_types
            FROM commits
            WHERE repo_id = :repo_id AND LOWER(author_name) = LOWER(:member_login)
            ORDER BY committer_date DESC;
        """)
        params = {"repo_id": repo_id, "member_login": member_login.lower()}
        result = self.db.execute(query, params).mappings().all()
        return [dict(row) for row in result]

    def _synthesize_insights(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Rút ra nhận xét từ dữ liệu thống kê."""
        insights = {}

        # 1. Xác định vai trò chính (Primary Role)
        area_dist = stats.get("area_distribution", {})
        if area_dist:
            primary_area = max(area_dist, key=area_dist.get)
            insights["primary_role"] = primary_area.replace("_", " ").title()
        else:
            insights["primary_role"] = "Generalist"

        # 2. Xác định phong cách làm việc (Work Style)
        type_dist = stats.get("type_distribution", {})
        if type_dist:
            total_commits = sum(type_dist.values())
            feat_ratio = type_dist.get("feat", 0) / total_commits if total_commits > 0 else 0
            fix_ratio = type_dist.get("fix", 0) / total_commits if total_commits > 0 else 0
            
            if feat_ratio > 0.5:
                insights["work_style"] = "Feature Developer"
            elif fix_ratio > 0.4:
                insights["work_style"] = "Dedicated Maintainer"
            elif type_dist.get("refactor", 0) > 20:
                insights["work_style"] = "Code Architect"
            else:
                insights["work_style"] = "Versatile Contributor"
        else:
            insights["work_style"] = "N/A"

        # 3. Hồ sơ chất lượng (Quality Profile)
        risk_dist = stats.get("risk_distribution", {})
        total_risk_commits = sum(risk_dist.values())
        high_risk_ratio = risk_dist.get("highrisk", 0) / total_risk_commits if total_risk_commits > 0 else 0
        
        if high_risk_ratio > 0.3:
            insights["quality_profile"] = "Handles Complex & High-Risk Tasks"
        elif high_risk_ratio < 0.05:
            insights["quality_profile"] = "Extremely Cautious & Consistent"
        else:
            insights["quality_profile"] = "Balanced Risk-Taker"

        # 4. Kỹ năng chính (Key Skills) - có thể mở rộng thêm
        key_skills = list(area_dist.keys())
        insights["key_skills"] = [skill.replace("_", " ").title() for skill in key_skills]

        return insights

    def generate_skill_profile(self, repo_id: int, member_login: str) -> Dict[str, Any]:
        """Tạo hồ sơ năng lực toàn diện cho một thành viên."""
        
        member_commits = self._get_member_commits(repo_id, member_login)
        
        if not member_commits:
            return {
                "member_login": member_login,
                "total_commits": 0,
                "message": "No commits found for this member in the repository."
            }

        # Chuẩn bị dữ liệu cho các model
        area_risk_inputs = []
        for commit in member_commits:
            area_risk_inputs.append({
                "commit_message": commit.get("message", ""),
                "diff_content": commit.get("diff_content", ""),
                "files_count": commit.get("files_changed", 0),
                "lines_added": commit.get("insertions", 0),
                "lines_removed": commit.get("deletions", 0),
                "total_changes": commit.get("insertions", 0) + commit.get("deletions", 0)
            })
        
        commit_type_inputs = self.commit_analyst_service._format_commits_for_ai(member_commits)

        # Gọi các model AI để phân tích
        predicted_areas = [self.area_analysis_service.predict_area(data) for data in area_risk_inputs]
        predicted_risks = [self.risk_analysis_service.predict_risk(data) for data in area_risk_inputs]
        predicted_commit_types_full = self.multifusion_service.predict_commit_type_batch(commit_type_inputs)
        predicted_commit_types = [res["commit_type"] for res in predicted_commit_types_full]

        # Tổng hợp kết quả
        stats = {
            "area_distribution": defaultdict(int),
            "risk_distribution": defaultdict(int),
            "type_distribution": defaultdict(int),
            "total_commits": len(member_commits),
            "total_lines_added": sum(c.get("insertions", 0) for c in member_commits),
            "total_lines_removed": sum(c.get("deletions", 0) for c in member_commits),
        }

        for area in predicted_areas:
            stats["area_distribution"][area] += 1
        for risk in predicted_risks:
            stats["risk_distribution"][risk] += 1
        for commit_type in predicted_commit_types:
            stats["type_distribution"][commit_type] += 1
            
        # Chuyển defaultdict sang dict thường để dễ serialize
        stats["area_distribution"] = dict(stats["area_distribution"])
        stats["risk_distribution"] = dict(stats["risk_distribution"])
        stats["type_distribution"] = dict(stats["type_distribution"])

        # Rút ra nhận xét
        insights = self._synthesize_insights(stats)

        return {
            "success": True,
            "member_login": member_login,
            "profile": {
                "statistics": stats,
                "insights": insights
            }
        }