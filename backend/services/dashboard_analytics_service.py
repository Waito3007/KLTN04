"""
Dashboard Analytics Service - Phân tích tiến độ, rủi ro và gợi ý phân công
Tuân thủ quy tắc KLTN04: Tách biệt logic, defensive programming, immutability
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from dataclasses import dataclass
import asyncio
import json
import logging
import os
import httpx
from statistics import mean, median

from services.multifusion_commitanalyst_service import MultiFusionV2Service
from services.area_analysis_service import AreaAnalysisService
from services.risk_analysis_service import RiskAnalysisService

logger = logging.getLogger(__name__)

@dataclass
class ProgressAnalysis:
    """Kết quả phân tích tiến độ"""
    total_commits: int
    commits_by_type: Dict[str, int]
    commits_by_area: Dict[str, int]
    commits_trend: List[Dict[str, Any]]
    velocity: float
    productivity_score: float
    recommendations: List[str]

@dataclass
class RiskAnalysis:
    """Kết quả phân tích rủi ro"""
    high_risk_commits: List[Dict[str, Any]]
    risk_trend: List[Dict[str, Any]]
    risk_score: float
    critical_areas: List[str]
    warnings: List[str]
    mitigation_suggestions: List[str]

@dataclass
class AssignmentSuggestion:
    """Gợi ý phân công công việc"""
    member_id: str
    member_name: str
    expertise_areas: List[str]
    suggested_tasks: List[Dict[str, Any]]
    workload_score: float
    availability_score: float
    skill_match_score: float
    rationale: str

class DashboardAnalyticsService:
    """Service chính cho dashboard analytics"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.commit_analyzer = MultiFusionV2Service()
        self.area_analyzer = AreaAnalysisService()
        self.risk_analyzer = RiskAnalysisService()
    
    async def get_comprehensive_analytics(
        self, 
        repo_owner: str, 
        repo_name: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Lấy phân tích toàn diện cho dashboard
        """
        try:
            # Lấy dữ liệu song song để tối ưu performance
            repo_id = await self._get_repo_id(repo_owner, repo_name)
            if not repo_id:
                raise ValueError(f"Repository {repo_owner}/{repo_name} not found")
            
            tasks = await asyncio.gather(
                self._analyze_progress(repo_id, days_back),
                self._analyze_risks(repo_id, days_back),
                self._generate_assignment_suggestions(repo_id, days_back),
                self._get_team_productivity_metrics(repo_id, days_back),
                return_exceptions=True
            )
            
            progress_analysis, risk_analysis, assignment_suggestions, productivity_metrics = tasks
            
            # Xử lý exceptions
            if isinstance(progress_analysis, Exception):
                logger.error(f"Progress analysis failed: {progress_analysis}")
                progress_analysis = self._get_empty_progress_analysis()
            
            if isinstance(risk_analysis, Exception):
                logger.error(f"Risk analysis failed: {risk_analysis}")
                risk_analysis = self._get_empty_risk_analysis()
            
            if isinstance(assignment_suggestions, Exception):
                logger.error(f"Assignment suggestions failed: {assignment_suggestions}")
                assignment_suggestions = []
            
            if isinstance(productivity_metrics, Exception):
                logger.error(f"Productivity metrics failed: {productivity_metrics}")
                productivity_metrics = {}
            
            # Try to synthesize higher-level, actionable recommendations using Gemini LLM
            llm_payload = None
            llm_result = None
            try:
                gemini_api_key = os.getenv('GEMINI_API_KEY')
                gemini_api_url = os.getenv('GEMINI_API_URL')
                if gemini_api_key and gemini_api_url:
                    prompt = {
                        "progress": {
                            "total_commits": progress_analysis.total_commits,
                            "velocity": progress_analysis.velocity,
                            "productivity_score": progress_analysis.productivity_score,
                            "commits_by_type": progress_analysis.commits_by_type,
                            "commits_by_area": progress_analysis.commits_by_area
                        },
                        "risks": {
                            "risk_score": risk_analysis.risk_score,
                            "critical_areas": risk_analysis.critical_areas,
                            "high_risk_commits_count": len(risk_analysis.high_risk_commits)
                        },
                        "assignments": [
                            {
                                "member_id": s.member_id,
                                "member_name": s.member_name,
                                "workload_score": s.workload_score,
                                "skill_match_score": s.skill_match_score,
                                "expertise": s.expertise_areas
                            } for s in assignment_suggestions
                        ],
                        "instructions": "Return a JSON object with keys: action_items (list of strings), prioritized_assignment_changes (list of {member_id, action, reason}), overall_recommendation (string)."
                    }

                    llm_payload = json.dumps(prompt)
                    # Call Gemini via helper
                    llm_raw = await self._call_gemini(llm_payload)
                    if llm_raw:
                        try:
                            llm_result = json.loads(llm_raw) if isinstance(llm_raw, str) else llm_raw
                        except Exception:
                            llm_result = {"overall_recommendation": str(llm_raw)}
            except Exception as e:
                logger.exception("LLM synthesis error: %s", e)

            # Merge LLM suggestions (if any) into progress recommendations
            progress_dict = progress_analysis.__dict__.copy()
            merged_recs = list(progress_dict.get('recommendations', []) or [])
            if llm_result and isinstance(llm_result.get('action_items'), list):
                merged_recs.extend([str(x) for x in llm_result.get('action_items')])
            elif llm_result and llm_result.get('overall_recommendation'):
                merged_recs.append(str(llm_result.get('overall_recommendation')))

            progress_dict['recommendations'] = merged_recs

            return {
                "repository": {
                    "owner": repo_owner,
                    "name": repo_name,
                    "id": repo_id
                },
                "analysis_period": {
                    "days": days_back,
                    "start_date": (datetime.now() - timedelta(days=days_back)).isoformat(),
                    "end_date": datetime.now().isoformat()
                },
                "progress": progress_dict,
                "risks": risk_analysis.__dict__,
                "assignment_suggestions": [s.__dict__ for s in assignment_suggestions],
                "productivity_metrics": productivity_metrics,  # Đã là dict đúng format
                "llm": llm_result,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Comprehensive analytics failed: {e}")
            raise
    
    async def _analyze_progress(self, repo_id: int, days_back: int) -> ProgressAnalysis:
        """Phân tích tiến độ dự án"""
        try:
            # Lấy commits trong khoảng thời gian
            commits_data = await self._get_commits_with_analysis(repo_id, days_back)
            
            if not commits_data:
                return self._get_empty_progress_analysis()
            
            # Phân tích theo loại commit
            commit_types = Counter(c.get('commit_type', 'unknown') for c in commits_data)
            
            # Phân tích theo khu vực
            commit_areas = Counter(c.get('dev_area', 'unknown') for c in commits_data)
            
            # Tính velocity (commits per day)
            velocity = len(commits_data) / days_back if days_back > 0 else 0
            
            # Tính productivity score
            productivity_score = self._calculate_productivity_score(commits_data)
            
            # Tạo trend data
            commits_trend = self._generate_commits_trend(commits_data, days_back)
            
            # Tạo recommendations
            recommendations = self._generate_progress_recommendations(
                commit_types, commit_areas, velocity, productivity_score
            )
            
            return ProgressAnalysis(
                total_commits=len(commits_data),
                commits_by_type=dict(commit_types),
                commits_by_area=dict(commit_areas),
                commits_trend=commits_trend,
                velocity=round(velocity, 2),
                productivity_score=round(productivity_score, 2),
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Progress analysis failed: {e}")
            return self._get_empty_progress_analysis()
    
    async def _analyze_risks(self, repo_id: int, days_back: int) -> RiskAnalysis:
        """Phân tích rủi ro dự án"""
        try:
            commits_data = await self._get_commits_with_analysis(repo_id, days_back)
            
            if not commits_data:
                return self._get_empty_risk_analysis()
            
            # Lọc commits có rủi ro cao
            high_risk_commits = [
                c for c in commits_data 
                if c.get('risk_level', '').lower() == 'highrisk'
            ]
            
            # Tính risk score tổng thể
            risk_score = len(high_risk_commits) / len(commits_data) if commits_data else 0
            
            # Phân tích trend rủi ro
            risk_trend = self._generate_risk_trend(commits_data, days_back)
            
            # Xác định khu vực critical
            critical_areas = self._identify_critical_areas(high_risk_commits)
            
            # Tạo warnings và suggestions
            warnings = self._generate_risk_warnings(high_risk_commits, risk_score)
            mitigation_suggestions = self._generate_mitigation_suggestions(
                high_risk_commits, critical_areas
            )
            
            return RiskAnalysis(
                high_risk_commits=high_risk_commits[:10],  # Top 10 rủi ro nhất
                risk_trend=risk_trend,
                risk_score=round(risk_score * 100, 2),
                critical_areas=critical_areas,
                warnings=warnings,
                mitigation_suggestions=mitigation_suggestions
            )
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return self._get_empty_risk_analysis()
    
    async def _generate_assignment_suggestions(
        self, 
        repo_id: int, 
        days_back: int
    ) -> List[AssignmentSuggestion]:
        """Tạo gợi ý phân công công việc"""
        try:
            # Lấy thông tin team members
            members = await self._get_team_members(repo_id)
            
            # Lấy tasks hiện tại
            current_tasks = await self._get_current_tasks(repo_id)
            
            # Phân tích skill của từng member
            member_skills = await self._analyze_member_skills(repo_id, members, days_back)
            
            suggestions = []
            
            for member in members:
                member_id = member.get('github_username', '')
                if not member_id:
                    continue
                
                # Lấy skill profile
                skills = member_skills.get(member_id, {})
                expertise_areas = skills.get('expertise_areas', [])
                
                # Tính workload hiện tại
                workload_score = self._calculate_workload_score(member_id, current_tasks)
                
                # Tính availability
                availability_score = self._calculate_availability_score(skills)
                
                # Suggest tasks phù hợp
                suggested_tasks = self._suggest_tasks_for_member(
                    member_id, expertise_areas, current_tasks, workload_score
                )
                
                # Tính skill match score
                skill_match_score = self._calculate_skill_match_score(
                    suggested_tasks, expertise_areas
                )
                
                # Tạo rationale
                rationale = self._generate_assignment_rationale(
                    expertise_areas, workload_score, availability_score, skill_match_score
                )
                
                suggestions.append(AssignmentSuggestion(
                    member_id=member_id,
                    member_name=member.get('name', member_id),
                    expertise_areas=expertise_areas,
                    suggested_tasks=suggested_tasks,
                    workload_score=round(workload_score, 2),
                    availability_score=round(availability_score, 2),
                    skill_match_score=round(skill_match_score, 2),
                    rationale=rationale
                ))
            
            # Sắp xếp theo tổng điểm phù hợp
            suggestions.sort(
                key=lambda s: (s.skill_match_score + s.availability_score - s.workload_score),
                reverse=True
            )
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Assignment suggestions failed: {e}")
            return []
    
    async def _get_team_productivity_metrics(
        self, 
        repo_id: int, 
        days_back: int
    ) -> Dict[str, Any]:
        """Lấy metrics về productivity của team"""
        try:
            # Lấy commits với phân tích
            commits_data = await self._get_commits_with_analysis(repo_id, days_back)
            
            if not commits_data:
                return {
                    "team_summary": {
                        "total_commits": 0,
                        "total_lines_changed": 0,
                        "average_commit_size": 0.0,
                        "fix_ratio": 0.0,
                        "active_contributors": 0
                    },
                    "member_metrics": {},
                    "trends": {
                        "daily_commits": [],
                        "weekly_velocity": 0.0
                    }
                }
            
            # Metrics theo thành viên
            member_metrics = defaultdict(lambda: {
                'commits': 0,
                'lines_added': 0,
                'lines_removed': 0,
                'files_changed': 0,
                'commit_types': Counter(),
                'areas': Counter()
            })
            
            for commit in commits_data:
                author = commit.get('author_name', 'unknown')
                member_metrics[author]['commits'] += 1
                
                # Xử lý an toàn cho các giá trị có thể None
                lines_added = commit.get('insertions') or 0
                lines_removed = commit.get('deletions') or 0
                files_changed = commit.get('files_changed') or 0
                
                member_metrics[author]['lines_added'] += lines_added
                member_metrics[author]['lines_removed'] += lines_removed
                member_metrics[author]['files_changed'] += files_changed
                member_metrics[author]['commit_types'][commit.get('commit_type', 'unknown')] += 1
                member_metrics[author]['areas'][commit.get('dev_area', 'unknown')] += 1
            
            # Convert Counter objects to dict
            for member in member_metrics:
                member_metrics[member]['commit_types'] = dict(member_metrics[member]['commit_types'])
                member_metrics[member]['areas'] = dict(member_metrics[member]['areas'])
            
            # Team-wide metrics với xử lý an toàn
            total_commits = len(commits_data)
            total_lines = sum((c.get('insertions') or 0) + (c.get('deletions') or 0) for c in commits_data)
            avg_commit_size = total_lines / total_commits if total_commits > 0 else 0
            
            # Code quality indicators
            fix_ratio = len([c for c in commits_data if c.get('commit_type') == 'fix']) / total_commits if total_commits > 0 else 0
            
            return {
                "team_summary": {
                    "total_commits": total_commits,
                    "total_lines_changed": total_lines,
                    "average_commit_size": round(avg_commit_size, 2),
                    "fix_ratio": round(fix_ratio * 100, 2),
                    "active_contributors": len(member_metrics)
                },
                "member_metrics": dict(member_metrics),
                "trends": {
                    "daily_commits": self._calculate_daily_trends(commits_data, days_back),
                    "weekly_velocity": self._calculate_weekly_velocity(commits_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Productivity metrics failed: {e}")
            # Trả về structure đầy đủ thay vì dict rỗng
            return {
                "team_summary": {
                    "total_commits": 0,
                    "total_lines_changed": 0,
                    "average_commit_size": 0.0,
                    "fix_ratio": 0.0,
                    "active_contributors": 0
                },
                "member_metrics": {},
                "trends": {
                    "daily_commits": [],
                    "weekly_velocity": 0.0
                }
            }
    
    # Helper methods
    async def _get_repo_id(self, repo_owner: str, repo_name: str) -> Optional[int]:
        """Lấy repo ID từ owner và name"""
        query = """
            SELECT id FROM repositories 
            WHERE owner = :owner AND name = :name
        """
        result = await self.db.execute(text(query), {
            "owner": repo_owner,
            "name": repo_name
        })
        row = result.fetchone()
        return row[0] if row else None
    
    async def _get_commits_with_analysis(self, repo_id: int, days_back: int) -> List[Dict[str, Any]]:
        """Lấy commits với phân tích AI"""
        query = """
            SELECT id, sha, message, author_name, committer_date, 
                   insertions, deletions, files_changed, modified_files, file_types,
                   diff_content
            FROM commits 
            WHERE repo_id = :repo_id 
            AND committer_date >= :start_date
            ORDER BY committer_date DESC
        """
        
        start_date = datetime.now() - timedelta(days=days_back)
        result = await self.db.execute(text(query), {
            "repo_id": repo_id,
            "start_date": start_date
        })
        
        commits = []
        for row in result.fetchall():
            commit_dict = dict(row._mapping)
            
            # Thêm AI analysis
            commit_dict['commit_type'] = await self._get_commit_type(commit_dict)
            commit_dict['dev_area'] = await self._get_dev_area(commit_dict)
            commit_dict['risk_level'] = await self._get_risk_level(commit_dict)
            
            commits.append(commit_dict)
        
        return commits
    
    async def _get_commit_type(self, commit_data: Dict[str, Any]) -> str:
        """Lấy loại commit từ MultiFusion model"""
        try:
            # Format data for MultiFusion model với xử lý an toàn cho None values
            lines_added = commit_data.get('insertions') or 0
            lines_removed = commit_data.get('deletions') or 0
            
            # Xử lý files_changed: từ string JSON sang list, sau đó lấy len
            files_changed_list = []
            if isinstance(commit_data.get('files_changed'), str):
                try:
                    files_changed_list = json.loads(commit_data['files_changed'])
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode files_changed JSON: {commit_data['files_changed']}")
            elif isinstance(commit_data.get('files_changed'), list):
                files_changed_list = commit_data['files_changed']
            
            num_files_changed = len(files_changed_list)
            
            formatted_commit = {
                'message': commit_data.get('message', ''),
                'file_count': num_files_changed,
                'lines_added': lines_added,
                'lines_removed': lines_removed,
                'total_changes': lines_added + lines_removed,
                'num_dirs_changed': 1  # Default value, có thể cải tiến sau
            }
            
            if self.commit_analyzer.is_model_available():
                results = self.commit_analyzer.predict_commit_type_batch([formatted_commit])
                if results and len(results) > 0:
                    return results[0].get('commit_type', 'unknown')
            
            return 'unknown'
        except Exception as e:
            logger.error(f"Commit type prediction failed: {e}")
            return 'unknown'
    
    async def _get_dev_area(self, commit_data: Dict[str, Any]) -> str:
        """Lấy khu vực phát triển từ Area Analysis model"""
        try:
            # Xử lý an toàn cho các giá trị None
            lines_added = commit_data.get('insertions') or 0
            lines_removed = commit_data.get('deletions') or 0
            
            # Xử lý files_changed: từ string JSON sang list, sau đó lấy len
            files_changed_list = []
            if isinstance(commit_data.get('files_changed'), str):
                try:
                    files_changed_list = json.loads(commit_data['files_changed'])
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode files_changed JSON for dev_area: {commit_data['files_changed']}")
            elif isinstance(commit_data.get('files_changed'), list):
                files_changed_list = commit_data['files_changed']
            
            num_files_changed = len(files_changed_list)
            
            formatted_data = {
                'commit_message': commit_data.get('message', ''),
                'diff_content': commit_data.get('diff_content', ''),
                'files_count': num_files_changed,
                'lines_added': lines_added,
                'lines_removed': lines_removed,
                'total_changes': lines_added + lines_removed
            }
            
            return self.area_analyzer.predict_area(formatted_data)
        except Exception as e:
            logger.error(f"Dev area prediction failed: {e}")
            return 'unknown'
    
    async def _get_risk_level(self, commit_data: Dict[str, Any]) -> str:
        """Lấy mức độ rủi ro từ Risk Analysis model"""
        try:
            # Xử lý an toàn cho các giá trị None
            lines_added = commit_data.get('insertions') or 0
            lines_removed = commit_data.get('deletions') or 0
            
            # Xử lý files_changed: từ string JSON sang list, sau đó lấy len
            files_changed_list = []
            if isinstance(commit_data.get('files_changed'), str):
                try:
                    files_changed_list = json.loads(commit_data['files_changed'])
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode files_changed JSON for risk_level: {commit_data['files_changed']}")
            elif isinstance(commit_data.get('files_changed'), list):
                files_changed_list = commit_data['files_changed']
            
            num_files_changed = len(files_changed_list)
            
            formatted_data = {
                'commit_message': commit_data.get('message', ''),
                'diff_content': commit_data.get('diff_content', ''),
                'files_count': num_files_changed,
                'lines_added': lines_added,
                'lines_removed': lines_removed,
                'total_changes': lines_added + lines_removed
            }
            
            return self.risk_analyzer.predict_risk(formatted_data)
        except Exception as e:
            logger.error(f"Risk prediction failed: {e}")
            return 'lowrisk'
    
    def _calculate_productivity_score(self, commits_data: List[Dict[str, Any]]) -> float:
        """Tính điểm productivity dựa trên chất lượng commits"""
        if not commits_data:
            return 0.0
        
        # Factors: commit frequency, code quality, diversity
        total_commits = len(commits_data)
        
        # Xử lý an toàn cho tính tổng changes
        total_changes = 0
        for c in commits_data:
            insertions = c.get('insertions') or 0
            deletions = c.get('deletions') or 0
            total_changes += insertions + deletions
            
        avg_commit_size = total_changes / total_commits if total_commits > 0 else 0
        
        # Penalize too large or too small commits
        size_score = max(0, 100 - abs(avg_commit_size - 50) * 2)
        
        # Reward diverse commit types
        commit_types = set(c.get('commit_type', 'unknown') for c in commits_data)
        diversity_score = min(len(commit_types) * 20, 100)
        
        # Overall score
        return (size_score + diversity_score) / 2
    
    def _generate_commits_trend(self, commits_data: List[Dict[str, Any]], days_back: int) -> List[Dict[str, Any]]:
        """Tạo dữ liệu trend commits theo ngày"""
        daily_commits = defaultdict(int)
        
        for commit in commits_data:
            commit_date = commit.get('committer_date')
            if commit_date:
                if isinstance(commit_date, str):
                    commit_date = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
                date_key = commit_date.date().isoformat()
                daily_commits[date_key] += 1
        
        # Tạo series đầy đủ cho tất cả ngày
        trend_data = []
        for i in range(days_back):
            date = (datetime.now() - timedelta(days=i)).date()
            trend_data.append({
                'date': date.isoformat(),
                'commits': daily_commits.get(date.isoformat(), 0)
            })
        
        return list(reversed(trend_data))
    
    def _generate_progress_recommendations(
        self, 
        commit_types: Counter, 
        commit_areas: Counter, 
        velocity: float, 
        productivity_score: float
    ) -> List[str]:
        """Tạo recommendations cho tiến độ"""
        recommendations = []
        
        if velocity < 1:
            recommendations.append("Tốc độ commit thấp. Khuyến nghị tăng tần suất commit nhỏ và thường xuyên.")
        
        if productivity_score < 50:
            recommendations.append("Điểm productivity thấp. Cân nhắc review quy trình và cải thiện chất lượng code.")
        
        fix_ratio = commit_types.get('fix', 0) / sum(commit_types.values()) if commit_types else 0
        if fix_ratio > 0.3:
            recommendations.append("Tỷ lệ commit fix cao. Cần cải thiện testing và code review.")
        
        if len(commit_areas) == 1:
            recommendations.append("Chỉ tập trung vào một khu vực. Khuyến nghị diversify development areas.")
        
        return recommendations
    
    def _get_empty_progress_analysis(self) -> ProgressAnalysis:
        """Trả về progress analysis rỗng"""
        return ProgressAnalysis(
            total_commits=0,
            commits_by_type={},
            commits_by_area={},
            commits_trend=[],
            velocity=0.0,
            productivity_score=0.0,
            recommendations=["Không có dữ liệu để phân tích tiến độ"]
        )
    
    def _get_empty_risk_analysis(self) -> RiskAnalysis:
        """Trả về risk analysis rỗng"""
        return RiskAnalysis(
            high_risk_commits=[],
            risk_trend=[],
            risk_score=0.0,
            critical_areas=[],
            warnings=["Không có dữ liệu để phân tích rủi ro"],
            mitigation_suggestions=[]
        )
    
    def _generate_risk_trend(self, commits_data: List[Dict[str, Any]], days_back: int) -> List[Dict[str, Any]]:
        """Tạo trend rủi ro theo ngày"""
        daily_risks = defaultdict(lambda: {'total': 0, 'high_risk': 0})
        
        for commit in commits_data:
            commit_date = commit.get('committer_date')
            if commit_date:
                if isinstance(commit_date, str):
                    commit_date = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
                date_key = commit_date.date().isoformat()
                daily_risks[date_key]['total'] += 1
                if commit.get('risk_level', '').lower() == 'highrisk':
                    daily_risks[date_key]['high_risk'] += 1
        
        trend_data = []
        for i in range(days_back):
            date = (datetime.now() - timedelta(days=i)).date()
            date_key = date.isoformat()
            risk_data = daily_risks.get(date_key, {'total': 0, 'high_risk': 0})
            risk_percentage = (risk_data['high_risk'] / risk_data['total'] * 100) if risk_data['total'] > 0 else 0
            
            trend_data.append({
                'date': date_key,
                'risk_percentage': round(risk_percentage, 2),
                'high_risk_commits': risk_data['high_risk'],
                'total_commits': risk_data['total']
            })
        
        return list(reversed(trend_data))
    
    def _identify_critical_areas(self, high_risk_commits: List[Dict[str, Any]]) -> List[str]:
        """Xác định khu vực có rủi ro cao"""
        area_risks = Counter()
        for commit in high_risk_commits:
            area = commit.get('dev_area', 'unknown')
            area_risks[area] += 1
        
        # Trả về top 3 khu vực rủi ro nhất
        return [area for area, _ in area_risks.most_common(3)]
    
    def _generate_risk_warnings(self, high_risk_commits: List[Dict[str, Any]], risk_score: float) -> List[str]:
        """Tạo cảnh báo rủi ro"""
        warnings = []
        
        if risk_score > 30:
            warnings.append(f"⚠️ Mức độ rủi ro cao ({risk_score:.1f}%). Cần review code kỹ lưỡng.")
        
        if len(high_risk_commits) > 10:
            warnings.append(f"⚠️ Có {len(high_risk_commits)} commits rủi ro cao trong kỳ.")
        
        # Cảnh báo về pattern rủi ro
        recent_risks = [c for c in high_risk_commits if self._is_recent_commit(c, 7)]
        if len(recent_risks) > 5:
            warnings.append("⚠️ Xu hướng tăng commits rủi ro cao trong 7 ngày qua.")
        
        return warnings
    
    def _generate_mitigation_suggestions(
        self, 
        high_risk_commits: List[Dict[str, Any]], 
        critical_areas: List[str]
    ) -> List[str]:
        """Tạo gợi ý giảm thiểu rủi ro"""
        suggestions = []
        
        if critical_areas:
            suggestions.append(f"🔍 Tăng cường code review cho các khu vực: {', '.join(critical_areas)}")
        
        large_commits = [c for c in high_risk_commits if (c.get('insertions', 0) + c.get('deletions', 0)) > 500]
        if large_commits:
            suggestions.append("📏 Khuyến nghị chia nhỏ commits lớn để dễ review và giảm rủi ro.")
        
        if len(high_risk_commits) > 0:
            suggestions.append("🧪 Tăng cường testing cho các thay đổi có rủi ro cao.")
            suggestions.append("👥 Assign senior developer review cho commits rủi ro cao.")
        
        return suggestions
    
    async def _get_team_members(self, repo_id: int) -> List[Dict[str, Any]]:
        """Lấy danh sách thành viên team"""
        query = """
            SELECT DISTINCT u.id, u.github_username, u.full_name, u.email
            FROM users u
            INNER JOIN commits c ON c.author_name = u.full_name OR c.author_name = u.github_username
            WHERE c.repo_id = :repo_id
            AND u.github_username IS NOT NULL
        """
        
        result = await self.db.execute(text(query), {"repo_id": repo_id})
        return [dict(row._mapping) for row in result.fetchall()]
    
    async def _get_current_tasks(self, repo_id: int) -> List[Dict[str, Any]]:
        """Lấy tasks hiện tại của repository"""
        query = """
            SELECT id, title, description, status, priority, assignee_github_username,
                   due_date, created_at, updated_at
            FROM project_tasks 
            WHERE repository_id = :repo_id
            AND status IN ('TODO', 'IN_PROGRESS')
            ORDER BY priority DESC, created_at ASC
        """
        
        result = await self.db.execute(text(query), {"repo_id": repo_id})
        return [dict(row._mapping) for row in result.fetchall()]
    
    async def _analyze_member_skills(
        self, 
        repo_id: int, 
        members: List[Dict[str, Any]], 
        days_back: int
    ) -> Dict[str, Dict[str, Any]]:
        """Phân tích kỹ năng của từng thành viên"""
        member_skills = {}
        
        for member in members:
            github_username = member.get('github_username')
            if not github_username:
                continue
            
            # Lấy commits của member
            query = """
                SELECT message, insertions, deletions, files_changed, modified_files, file_types,
                       diff_content, committer_date
                FROM commits 
                WHERE repo_id = :repo_id 
                AND (author_name = :username OR author_name = :name)
                AND committer_date >= :start_date
                ORDER BY committer_date DESC
            """
            
            start_date = datetime.now() - timedelta(days=days_back)
            result = await self.db.execute(text(query), {
                "repo_id": repo_id,
                "username": github_username,
                "name": member.get('name', ''),
                "start_date": start_date
            })
            
            member_commits = [dict(row._mapping) for row in result.fetchall()]
            
            # Phân tích skills
            skills = await self._extract_member_skills(member_commits, days_back)
            member_skills[github_username] = skills
        
        return member_skills
    
    async def _extract_member_skills(self, commits: List[Dict[str, Any]], days_back: int) -> Dict[str, Any]:
        """Trích xuất kỹ năng từ commits của member"""
        if not commits:
            return {
                'expertise_areas': [], 
                'skill_level': 'junior', 
                'specializations': [],
                'last_commit_date': None
            }
        
        # Phân tích areas
        areas = []
        for commit in commits:
            area = await self._get_dev_area(commit)
            if area != 'unknown':
                areas.append(area)
        
        area_counts = Counter(areas)
        expertise_areas = [area for area, count in area_counts.most_common(3)]
        
        # Phân tích file types để xác định specializations
        file_types = []
        for commit in commits:
            file_types_data = commit.get('file_types', {})
            if isinstance(file_types_data, str):
                try:
                    file_types_data = json.loads(file_types_data)
                except:
                    file_types_data = {}
            
            if isinstance(file_types_data, dict):
                for file_type in file_types_data.keys():
                    file_types.append(file_type.lstrip('.'))
        
        type_counts = Counter(file_types)
        specializations = [ft for ft, count in type_counts.most_common(3) if count > 2]
        
        # Estimate skill level based on commit patterns - xử lý an toàn cho None values
        commit_sizes = []
        for c in commits:
            insertions = c.get('insertions') or 0
            deletions = c.get('deletions') or 0
            commit_sizes.append(insertions + deletions)
            
        avg_commit_size = mean(commit_sizes) if commit_sizes else 0
        commit_frequency = len(commits) / days_back if days_back > 0 else 0
        
        skill_level = 'junior'
        if avg_commit_size > 100 and commit_frequency > 1:
            skill_level = 'senior'
        elif avg_commit_size > 50 or commit_frequency > 0.5:
            skill_level = 'mid'
        
        # Lấy ngày commit cuối cùng
        commit_dates = [c.get('committer_date') for c in commits if c.get('committer_date')]
        last_commit_date = max(commit_dates) if commit_dates else None

        return {
            'expertise_areas': expertise_areas,
            'skill_level': skill_level,
            'specializations': specializations,
            'avg_commit_size': round(avg_commit_size, 2),
            'commit_frequency': round(commit_frequency, 2),
            'last_commit_date': last_commit_date
        }
    
    def _calculate_workload_score(self, member_id: str, current_tasks: List[Dict[str, Any]]) -> float:
        """Tính điểm workload hiện tại của member"""
        member_tasks = [t for t in current_tasks if t.get('assignee_github_username') == member_id]
        
        if not member_tasks:
            return 0.0
        
        # Scoring based on number and priority of tasks
        workload = 0
        priority_weights = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        
        for task in member_tasks:
            priority = task.get('priority', 'MEDIUM')
            weight = priority_weights.get(priority, 2)
            workload += weight
        
        return min(workload / 10 * 100, 100)  # Normalize to 0-100
    
    def _calculate_availability_score(self, member_skills: Dict[str, Any]) -> float:
        """Tính điểm availability của member dựa trên hoạt động gần đây."""
        last_commit_date = member_skills.get('last_commit_date')

        if not last_commit_date:
            return 20.0  # Rất thấp nếu không có commit nào

        if isinstance(last_commit_date, str):
            last_commit_date = datetime.fromisoformat(last_commit_date.replace('Z', '+00:00'))

        days_since_last_commit = (datetime.now() - last_commit_date).days

        if days_since_last_commit <= 2:
            return 95.0  # Rất sẵn sàng
        elif days_since_last_commit <= 7:
            return 75.0  # Sẵn sàng
        elif days_since_last_commit <= 14:
            return 50.0  # Ít hoạt động
        else:
            return 30.0  # Không hoạt động gần đây
    
    def _suggest_tasks_for_member(
        self, 
        member_id: str, 
        expertise_areas: List[str], 
        current_tasks: List[Dict[str, Any]],
        workload_score: float
    ) -> List[Dict[str, Any]]:
        """Gợi ý tasks phù hợp cho member"""
        if workload_score > 80:  # Quá tải
            return []
        
        # Filter unassigned tasks that match expertise
        unassigned_tasks = [t for t in current_tasks if not t.get('assignee_github_username')]
        
        suggested_tasks = []
        for task in unassigned_tasks[:5]:  # Top 5 suggestions
            # Simple matching based on task title/description keywords
            task_text = f"{task.get('title', '')} {task.get('description', '')}".lower()
            
            match_score = 0
            for area in expertise_areas:
                if area.lower() in task_text:
                    match_score += 1
            
            if match_score > 0 or len(expertise_areas) == 0:
                suggested_tasks.append({
                    'task_id': task.get('id'),
                    'title': task.get('title'),
                    'priority': task.get('priority'),
                    'match_score': match_score,
                    'estimated_effort': self._estimate_task_effort(task)
                })
        
        return sorted(suggested_tasks, key=lambda x: x['match_score'], reverse=True)
    
    def _calculate_skill_match_score(
        self, 
        suggested_tasks: List[Dict[str, Any]], 
        expertise_areas: List[str]
    ) -> float:
        """Tính điểm match giữa skills và suggested tasks"""
        if not suggested_tasks:
            return 0.0
        
        total_match = sum(task.get('match_score', 0) for task in suggested_tasks)
        max_possible = len(suggested_tasks) * len(expertise_areas)
        
        return (total_match / max_possible * 100) if max_possible > 0 else 0.0
    
    def _generate_assignment_rationale(
        self, 
        expertise_areas: List[str], 
        workload_score: float, 
        availability_score: float, 
        skill_match_score: float
    ) -> str:
        """Tạo lý do cho assignment suggestion"""
        reasons = []
        
        if skill_match_score > 70:
            reasons.append(f"Chuyên môn phù hợp cao ({skill_match_score:.0f}%)")
        
        if workload_score < 50:
            reasons.append("Khối lượng công việc hiện tại thấp")
        elif workload_score > 80:
            reasons.append("⚠️ Khối lượng công việc cao, cần cân nhắc")
        
        if availability_score > 70:
            reasons.append("Tính sẵn sàng cao")
        
        if expertise_areas:
            reasons.append(f"Chuyên về: {', '.join(expertise_areas)}")
        
        return "; ".join(reasons) if reasons else "Phân công cân bằng workload team"
    
    def _estimate_task_effort(self, task: Dict[str, Any]) -> str:
        """Ước lượng effort cho task"""
        priority = task.get('priority', 'MEDIUM')
        description_length = len(task.get('description', ''))
        
        if priority == 'CRITICAL' or description_length > 500:
            return 'HIGH'
        elif priority == 'HIGH' or description_length > 200:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _is_recent_commit(self, commit: Dict[str, Any], days: int) -> bool:
        """Kiểm tra commit có trong khoảng thời gian gần đây không"""
        commit_date = commit.get('committer_date')
        if not commit_date:
            return False
        
        if isinstance(commit_date, str):
            commit_date = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
        
        return (datetime.now() - commit_date).days <= days
    
    def _calculate_daily_trends(self, commits_data: List[Dict[str, Any]], days_back: int) -> List[Dict[str, Any]]:
        """Tính trends theo ngày"""
        daily_stats = defaultdict(lambda: {'commits': 0, 'lines': 0})
        
        for commit in commits_data:
            commit_date = commit.get('committer_date')
            if commit_date:
                if isinstance(commit_date, str):
                    commit_date = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
                date_key = commit_date.date().isoformat()
                daily_stats[date_key]['commits'] += 1
                
                # Xử lý an toàn cho None values
                insertions = commit.get('insertions') or 0
                deletions = commit.get('deletions') or 0
                daily_stats[date_key]['lines'] += insertions + deletions
        
        trends = []
        for i in range(days_back):  # Use full days_back period
            date = (datetime.now() - timedelta(days=i)).date()
            date_key = date.isoformat()
            stats = daily_stats.get(date_key, {'commits': 0, 'lines': 0})
            trends.append({
                'date': date_key,
                'commits': stats['commits'],
                'lines_changed': stats['lines']
            })
        
        return list(reversed(trends))
    
    def _calculate_weekly_velocity(self, commits_data: List[Dict[str, Any]]) -> float:
        """Tính velocity theo tuần"""
        if not commits_data:
            return 0.0
        
        # Group by week
        weekly_commits = defaultdict(int)
        for commit in commits_data:
            commit_date = commit.get('committer_date')
            if commit_date:
                if isinstance(commit_date, str):
                    commit_date = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
                # Get ISO week
                week_key = f"{commit_date.year}-W{commit_date.isocalendar()[1]:02d}"
                weekly_commits[week_key] += 1
        
        if not weekly_commits:
            return 0.0
        
        return mean(weekly_commits.values())

    async def _call_gemini(self, prompt_json: str) -> Optional[str]:
        """Call Gemini LLM endpoint. Expect prompt_json to be a JSON-stringified object.
        Returns raw text (often JSON) or None on failure.
        Requires GEMINI_API_KEY and GEMINI_API_URL in env.
        """
        api_key = os.getenv('GEMINI_API_KEY')
        api_url = os.getenv('GEMINI_API_URL')
        if not api_key or not api_url:
            logger.debug("Gemini API not configured; skipping LLM call")
            return None

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            'input': json.loads(prompt_json),
            'max_output_tokens': 512
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(api_url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                # Try common keys
                if isinstance(data, dict):
                    if 'output' in data:
                        return data['output']
                    if 'text' in data:
                        return data['text']
                    if 'response' in data:
                        return data['response']
                    if 'choices' in data and isinstance(data['choices'], list) and data['choices']:
                        first = data['choices'][0]
                        return first.get('message') or first.get('text')
                return json.dumps(data)
        except Exception as e:
            logger.exception("Gemini call failed: %s", e)
            return None

    async def get_developer_dna(self, repo_id: int, author_name: str, days_back: int, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Phân tích và xây dựng hồ sơ DNA của một developer.
        Luôn gọi trực tiếp các model AI để có output chuẩn đoán.
        """
        # 1. Lấy commits của author (chỉ các cột cơ bản)
        query = """
            SELECT id, sha, message, author_name, committer_date, insertions, deletions, 
                   files_changed, modified_files, file_types, diff_content
            FROM commits 
            WHERE repo_id = :repo_id 
            AND author_name = :author_name
            AND committer_date >= :start_date
        """
        start_date = datetime.now() - timedelta(days=days_back)
        result = await self.db.execute(text(query), {
            "repo_id": repo_id,
            "author_name": author_name,
            "start_date": start_date
        })
        author_commits = [dict(row._mapping) for row in result.fetchall()]

        if not author_commits:
            return {"message": "Không có đủ dữ liệu cho developer này."}

        # Chuyển đổi committer_date sang datetime object nếu cần và chạy AI analysis
        analyzed_commits = []
        for commit in author_commits:
            # Chuyển đổi committer_date
            if isinstance(commit['committer_date'], str):
                try:
                    commit['committer_date'] = datetime.fromisoformat(commit['committer_date'].replace('Z', '+00:00'))
                except ValueError:
                    logger.warning(f"Could not parse committer_date string: {commit['committer_date']}")
                    continue # Bỏ qua commit này nếu ngày không hợp lệ
            
            # Luôn chạy AI analysis cho mỗi commit
            commit['commit_type'] = await self._get_commit_type(commit)
            commit['dev_area'] = await self._get_dev_area(commit)
            commit['risk_level'] = await self._get_risk_level(commit)
            analyzed_commits.append(commit)

        # Lọc ra các commit hợp lệ sau khi xử lý ngày
        valid_analyzed_commits = [c for c in analyzed_commits if isinstance(c['committer_date'], datetime)]

        # 4. Phân tích các "Gen"
        # Gen 1: Nhịp độ & Mẫu hình làm việc
        hours = [c['committer_date'].hour for c in valid_analyzed_commits]
        weekdays = [c['committer_date'].weekday() for c in valid_analyzed_commits]
        work_rhythm = {
            "active_hours": dict(Counter(hours).most_common(5)),
            "active_days": dict(Counter(weekdays).most_common(3)),
            "total_commits": len(valid_analyzed_commits),
            "commit_frequency": round(len(valid_analyzed_commits) / days_back, 2) if days_back > 0 else 0
        }

        # Gen 2: Phong cách đóng góp
        commit_types = Counter(c.get('commit_type', 'unknown') for c in valid_analyzed_commits)
        contribution_style = {
            "distribution": dict(commit_types),
            "primary_style": commit_types.most_common(1)[0][0] if commit_types else 'unknown'
        }

        # Gen 3: Chuyên môn kỹ thuật
        tech_expertise = {
            "areas": Counter(c.get('dev_area', 'unknown') for c in valid_analyzed_commits),
            "languages": Counter()
        }
        for c in valid_analyzed_commits:
            try:
                file_types = json.loads(c.get('file_types', '{}')) if isinstance(c.get('file_types'), str) else c.get('file_types', {})
                if file_types:
                    tech_expertise["languages"].update(file_types)
            except (json.JSONDecodeError, TypeError):
                continue
        
        tech_expertise["areas"] = dict(tech_expertise["areas"].most_common(3))
        tech_expertise["languages"] = dict(tech_expertise["languages"].most_common(5))

        # Gen 4: Hồ sơ rủi ro và chất lượng
        risk_levels = Counter(c.get('risk_level', 'lowrisk') for c in valid_analyzed_commits)
        total_risk_commits = sum(risk_levels.values())
        high_risk_pct = (risk_levels.get('highrisk', 0) / total_risk_commits * 100) if total_risk_commits > 0 else 0
        
        commit_sizes = [(c.get('insertions',0) or 0) + (c.get('deletions',0) or 0) for c in valid_analyzed_commits]
        
        risk_profile = {
            "distribution": dict(risk_levels),
            "high_risk_percentage": round(high_risk_pct, 2),
            "avg_commit_size": round(mean(commit_sizes) if commit_sizes else 0, 2),
            "max_commit_size": max(commit_sizes) if commit_sizes else 0,
        }

        return {
            "author_name": author_name,
            "analysis_period_days": days_back,
            "work_rhythm": work_rhythm,
            "contribution_style": contribution_style,
            "tech_expertise": tech_expertise,
            "risk_profile": risk_profile
        }
