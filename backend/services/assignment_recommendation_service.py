# backend/services/assignment_recommendation_service.py
"""
Assignment Recommendation Service - Đề xuất phân công thành viên dựa trên phân tích commit history
Sử dụng AI analysis results từ MultiFusion V2 và member analysis để đề xuất người phù hợp
"""

from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import math
from services.multifusion_commitanalyst_service import MultifusionCommitAnalystService
from services.area_analysis_service import AreaAnalysisService
from services.risk_analysis_service import RiskAnalysisService

logger = logging.getLogger(__name__)

class MemberSkillProfileService:
    """
    Service tạo hồ sơ kỹ năng thành viên dựa trên kết quả phân tích AI commit type, area, risk.
    """
    def __init__(self, db: Session):
        self.db = db
        self.commit_analyst = MultifusionCommitAnalystService(db)
        self.area_analyst = AreaAnalysisService()
        self.risk_analyst = RiskAnalysisService()

    def build_member_skill_profiles(self, repository_id: int, days_back: int = 90) -> Dict[str, Dict[str, Any]]:
        """
        Tạo hồ sơ kỹ năng cho từng thành viên dựa trên kết quả phân tích AI.
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        query = text("""
            SELECT 
                c.author_name,
                c.message,
                c.insertions,
                c.deletions,
                c.files_changed,
                c.modified_files,
                c.committer_date,
                c.branch_name,
                c.diff_content
            FROM commits c
            WHERE c.repo_id = :repo_id 
                AND c.committer_date >= :cutoff_date
            ORDER BY c.committer_date DESC
        """)
        commits_data = self.db.execute(query, {
            "repo_id": repository_id,
            "cutoff_date": cutoff_date
        }).fetchall()

        member_profiles = defaultdict(lambda: {
            'commit_types': defaultdict(int),
            'areas': defaultdict(int),
            'risk_levels': defaultdict(int),
            'total_commits': 0,
            'total_changes': 0,
            'recent_activity_score': 0.0,
            'expertise_areas': [],
            'risk_tolerance': 'low',
            'ai_coverage': 0.0
        })

        for commit in commits_data:
            author = commit[0]
            message = commit[1]
            insertions = commit[2] or 0
            deletions = commit[3] or 0
            files_changed = commit[4] or 0
            modified_files = commit[5]
            commit_date = commit[6]
            diff_content = commit[8] or ''

            # Chuẩn bị dữ liệu cho các service phân tích
            commit_data_for_ai = {
                'message': message,
                'diff_content': diff_content,
                'lines_added': insertions,
                'lines_removed': deletions,
                'file_count': files_changed,
                'total_changes': insertions + deletions
            }

            # Phân tích AI
            commit_type = None
            area = None
            risk = None
            try:
                # Commit type
                ai_result = self.commit_analyst.multifusion_v2_service.predict_commit_type_batch([commit_data_for_ai])
                commit_type = ai_result[0]['commit_type'] if ai_result else 'other'
            except Exception as e:
                commit_type = 'other'
            try:
                area = self.area_analyst.predict_area({
                    'commit_message': message,
                    'diff_content': diff_content,
                    'files_count': files_changed,
                    'lines_added': insertions,
                    'lines_removed': deletions,
                    'total_changes': insertions + deletions
                })
            except Exception as e:
                area = 'general'
            try:
                risk = self.risk_analyst.predict_risk({
                    'commit_message': message,
                    'diff_content': diff_content,
                    'files_count': files_changed,
                    'lines_added': insertions,
                    'lines_removed': deletions,
                    'total_changes': insertions + deletions
                })
            except Exception as e:
                risk = 'low'

            profile = member_profiles[author]
            profile['commit_types'][commit_type] += 1
            profile['areas'][area] += 1
            profile['risk_levels'][risk] += 1
            profile['total_commits'] += 1
            profile['total_changes'] += insertions + deletions
            # Tính recent activity score (commits gần đây có trọng số cao hơn)
            days_ago = (datetime.now() - commit_date).days if commit_date else days_back
            recency_weight = max(0.1, 1.0 - (days_ago / days_back))
            profile['recent_activity_score'] += recency_weight

        # Tổng hợp expertise area và risk tolerance
        for author, profile in member_profiles.items():
            if profile['total_commits'] > 0:
                sorted_areas = sorted(profile['areas'].items(), key=lambda x: x[1], reverse=True)
                profile['expertise_areas'] = [area for area, count in sorted_areas[:2] if count >= 2]
                total_risk = sum(profile['risk_levels'].values())
                high_risk = profile['risk_levels'].get('highrisk', 0) + profile['risk_levels'].get('high', 0)
                ratio = high_risk / total_risk if total_risk > 0 else 0
                if ratio > 0.3:
                    profile['risk_tolerance'] = 'high'
                elif ratio > 0.1:
                    profile['risk_tolerance'] = 'medium'
                else:
                    profile['risk_tolerance'] = 'low'
                profile['ai_coverage'] = 1.0  # Vì toàn bộ commit đều dùng AI

        return dict(member_profiles)
    
    def recommend_assignees(
        self, 
        repository_id: int, 
        task_type: str, 
        task_area: str, 
        risk_level: str,
        required_skills: Optional[List[str]] = None,
        exclude_members: Optional[List[str]] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Đề xuất thành viên phù hợp cho một task cụ thể
        
        Args:
            repository_id: ID của repository
            task_type: Loại task (feat, fix, docs, refactor, etc.)
            task_area: Phạm vi task (frontend, backend, database, etc.)
            risk_level: Mức độ rủi ro (low, medium, high)
            required_skills: Danh sách kỹ năng yêu cầu (optional)
            exclude_members: Danh sách members không xem xét (optional)
            top_k: Số lượng đề xuất tối đa
            
        Returns:
            List các đề xuất với score và lý do
        """
        # SỬA: Gọi đúng hàm build_member_skill_profiles
        member_skills = self.build_member_skill_profiles(repository_id)
        
        if not member_skills:
            return []
        
        recommendations = []
        
        for member, profile in member_skills.items():
            if exclude_members and member in exclude_members:
                continue
                
            if profile['total_commits'] == 0:
                continue
            
            # Tính matching score
            score = self._calculate_matching_score(
                profile, task_type, task_area, risk_level, required_skills
            )
            
            # Tạo explanation
            explanation = self._generate_explanation(
                profile, task_type, task_area, risk_level
            )
            
            recommendations.append({
                'member': member,
                'score': round(score, 2),
                'explanation': explanation,
                'profile_summary': {
                    'total_commits': profile['total_commits'],
                    'expertise_areas': profile['expertise_areas'],
                    'risk_tolerance': profile['risk_tolerance'],
                    'recent_activity_score': round(profile['recent_activity_score'], 2),
                    'top_commit_types': dict(sorted(
                        profile['commit_types'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:3])
                }
            })
        
        # Sắp xếp theo score và trả về top k
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]
    
    def _calculate_matching_score(
        self, 
        profile: Dict[str, Any], 
        task_type: str, 
        task_area: str, 
        risk_level: str,
        required_skills: Optional[List[str]] = None
    ) -> float:
        """Tính toán điểm phù hợp cho một member dựa trên AI analysis"""
        score = 0.0
        
        # Determine whether to use AI predictions or legacy analysis
        use_ai = profile.get('ai_coverage', 0) > 0.5  # Use AI if >50% commits analyzed by AI
        
        # SỬA: Bỏ điều kiện `and 'ai_predictions' in profile` và dùng trực tiếp profile data
        if use_ai:
            # Use AI predictions for scoring
            ai_commit_types = profile['commit_types']
            ai_areas = profile['areas']
            ai_risks = profile['risk_levels']
            
            # 1. Task type experience (30% of total score) - AI enhanced
            ai_task_type_count = ai_commit_types.get(task_type, 0)
            total_ai_commits = sum(ai_commit_types.values())
            if total_ai_commits > 0:
                ai_task_type_ratio = ai_task_type_count / total_ai_commits
                score += ai_task_type_ratio * 30
                logger.debug(f"AI Task type score: {ai_task_type_ratio * 30}")
            
            # 2. Area expertise (35% of total score) - AI enhanced
            ai_area_count = ai_areas.get(task_area, 0)
            # Also check for related areas
            related_areas = self._get_related_areas(task_area)
            for related_area in related_areas:
                ai_area_count += ai_areas.get(related_area, 0) * 0.5  # 50% weight for related areas
                
            if total_ai_commits > 0:
                ai_area_ratio = min(1.0, ai_area_count / total_ai_commits)  # Cap at 1.0
                score += ai_area_ratio * 35
                logger.debug(f"AI Area score: {ai_area_ratio * 35}")
            
            # 3. Risk tolerance match (25% of total score) - AI enhanced
            ai_risk_tolerance = self._calculate_ai_risk_tolerance(ai_risks)
            risk_match_score = self._get_risk_match_score(ai_risk_tolerance, risk_level)
            score += risk_match_score * 25
            logger.debug(f"AI Risk score: {risk_match_score * 25}")
            
        else:
            # Fallback to legacy analysis (giữ nguyên logic này phòng trường hợp ai_coverage thấp)
            total_commits = profile['total_commits']
            
            # 1. Task type experience (25% of total score)
            task_type_count = profile['commit_types'].get(task_type, 0)
            if total_commits > 0:
                task_type_ratio = task_type_count / total_commits
                score += task_type_ratio * 25
            
            # 2. Area expertise (30% of total score)
            area_count = profile['areas'].get(task_area, 0)
            if total_commits > 0:
                area_ratio = area_count / total_commits
                score += area_ratio * 30
            
            # 3. Risk tolerance match (20% of total score)
            risk_tolerance = profile['risk_tolerance']
            risk_match_score = self._get_risk_match_score(risk_tolerance, risk_level)
            score += risk_match_score * 20
        
        # 4. Recent activity (10% of total score)
        activity_score = min(1.0, profile['recent_activity_score'] / 10)  # Normalize
        score += activity_score * 10
        
        # 5. Overall experience (5% of total score, giảm trọng số)
        total_commits = profile['total_commits']
        experience_score = min(1.0, total_commits / 50)  # Normalize với max 50 commits
        score += experience_score * 5
        
        # SỬA: Tạm thời loại bỏ bonus này vì profile không có 'languages'
        # if required_skills:
        #     skill_bonus = 0
        #     for skill in required_skills:
        #         if skill in profile['languages'] and profile['languages'][skill] > 0:
        #             skill_bonus += 5
        #     score += min(skill_bonus, 15)  # Max 15 bonus points
        
        # AI analysis bonus
        if use_ai:
            score += 5  # 5 point bonus for AI-enhanced analysis
        
        return score
    
    def _get_risk_match_score(self, member_tolerance: str, task_risk: str) -> float:
        """Tính điểm phù hợp về risk tolerance"""
        risk_matrix = {
            ('high', 'high'): 1.0,
            ('high', 'medium'): 0.8,
            ('high', 'low'): 0.6,
            ('medium', 'high'): 0.7,
            ('medium', 'medium'): 1.0,
            ('medium', 'low'): 0.9,
            ('low', 'high'): 0.3,
            ('low', 'medium'): 0.7,
            ('low', 'low'): 1.0,
        }
        return risk_matrix.get((member_tolerance, task_risk), 0.5)
    
    def _get_related_areas(self, task_area: str) -> List[str]:
        """Lấy danh sách areas liên quan để mở rộng matching"""
        related_map = {
            'frontend': ['ui', 'interface', 'web'],
            'backend': ['api', 'server', 'service'],
            'database': ['data', 'storage', 'sql'],
            'devops': ['deployment', 'infrastructure', 'ci'],
            'mobile': ['android', 'ios', 'app'],
            'docs': ['documentation', 'readme'],
            'testing': ['test', 'qa', 'spec']
        }
        return related_map.get(task_area, [])
    
    def _calculate_ai_risk_tolerance(self, ai_risks: Dict[str, int]) -> str:
        """Tính toán risk tolerance từ AI predictions"""
        total_ai_commits = sum(ai_risks.values())
        if total_ai_commits == 0:
            return 'low'
        
        high_risk_count = ai_risks.get('highrisk', 0) + ai_risks.get('high', 0)
        high_risk_ratio = high_risk_count / total_ai_commits
        
        if high_risk_ratio > 0.4:
            return 'high'
        elif high_risk_ratio > 0.15:
            return 'medium'
        else:
            return 'low'
    
    def _generate_explanation(
        self, 
        profile: Dict[str, Any], 
        task_type: str, 
        task_area: str, 
        risk_level: str
    ) -> str:
        """Tạo explanation cho đề xuất với thông tin AI analysis"""
        explanations = []
        
        # Determine if using AI analysis
        use_ai = profile.get('ai_coverage', 0) > 0.5
        
        # SỬA: Bỏ điều kiện `and 'ai_predictions' in profile` và dùng trực tiếp profile data
        if use_ai:
            # AI-enhanced explanations
            ai_commit_types = profile['commit_types']
            ai_areas = profile['areas']
            ai_risks = profile['risk_levels']
            
            # Experience với task type (AI)
            task_type_count = ai_commit_types.get(task_type, 0)
            if task_type_count > 0:
                explanations.append(f"AI phân tích: {task_type_count} commits loại '{task_type}'")
            
            # Experience với area (AI)
            area_count = ai_areas.get(task_area, 0)
            if area_count > 0:
                explanations.append(f"AI phân tích: {area_count} commits trong '{task_area}'")
            
            # Risk tolerance (AI)
            ai_risk_tolerance = self._calculate_ai_risk_tolerance(ai_risks)
            if ai_risk_tolerance == 'high' and risk_level == 'high':
                explanations.append("AI xác nhận: Chuyên xử lý tasks rủi ro cao")
            elif ai_risk_tolerance == 'low' and risk_level == 'low':
                explanations.append("AI xác nhận: Chuyên xử lý tasks ổn định")
                
        else:
            # Legacy explanations
            # Experience với task type
            task_type_count = profile['commit_types'].get(task_type, 0)
            if task_type_count > 0:
                explanations.append(f"Có {task_type_count} commits loại '{task_type}'")
            
            # Experience với area
            area_count = profile['areas'].get(task_area, 0)
            if area_count > 0:
                explanations.append(f"Có {area_count} commits trong '{task_area}'")
            
            # Risk tolerance
            risk_tolerance = profile['risk_tolerance']
            if risk_tolerance == 'high' and risk_level == 'high':
                explanations.append("Thường xuyên xử lý tasks có độ rủi ro cao")
            elif risk_tolerance == 'low' and risk_level == 'low':
                explanations.append("Chuyên xử lý tasks ổn định, ít rủi ro")
        
        # Common explanations
        # Recent activity
        if profile['recent_activity_score'] > 5:
            explanations.append("Hoạt động tích cực gần đây")
        
        # Expertise areas
        if task_area in profile['expertise_areas']:
            explanations.append(f"Là chuyên gia trong lĩnh vực '{task_area}'")
        
        return "; ".join(explanations) if explanations else "Đề xuất dựa trên phân tích tổng thể."

    def get_member_workload(self, repository_id: int, days_back: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Phân tích workload hiện tại của các thành viên
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Query active tasks
            task_query = text("""
                SELECT 
                    pt.assignee_github_username,
                    pt.status,
                    pt.priority,
                    COUNT(*) as task_count
                FROM project_tasks pt
                WHERE pt.repository_id = :repo_id 
                    AND pt.status IN ('TODO', 'IN_PROGRESS')
                    AND pt.created_at >= :cutoff_date
                    AND pt.assignee_github_username IS NOT NULL
                GROUP BY pt.assignee_github_username, pt.status, pt.priority
            """)
            
            task_results = self.db.execute(task_query, {
                "repo_id": repository_id,
                "cutoff_date": cutoff_date
            }).fetchall()
            
            logger.info(f"Found {len(task_results)} task workload records for repository {repository_id}")
            
            member_workload = defaultdict(lambda: {
                'active_tasks': 0,
                'in_progress_tasks': 0,
                'high_priority_tasks': 0,
                'workload_score': 0.0
            })
            
            for result in task_results:
                member = result[0]
                if not member:
                    continue
                    
                status = result[1]
                priority = result[2]
                count = result[3]
                
                workload = member_workload[member]
                workload['active_tasks'] += count
                
                if status == 'IN_PROGRESS':
                    workload['in_progress_tasks'] += count
                
                if priority in ['HIGH', 'URGENT']:
                    workload['high_priority_tasks'] += count
                
                # Calculate workload score
                priority_weight = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'URGENT': 4}
                status_weight = {'TODO': 1, 'IN_PROGRESS': 1.5}
                
                weight = priority_weight.get(priority, 1) * status_weight.get(status, 1)
                workload['workload_score'] += count * weight
            
            return dict(member_workload)
            
        except Exception as e:
            logger.error(f"Error getting member workload for repository {repository_id}: {str(e)}")
            return {}

    def recommend_with_workload_balance(
        self,
        repository_id: int,
        task_type: str,
        task_area: str, 
        risk_level: str,
        task_priority: str = 'MEDIUM',
        required_skills: Optional[List[str]] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Đề xuất phân công có tính đến workload balancing
        """
        try:
            # Lấy recommendations cơ bản
            basic_recommendations = self.recommend_assignees(
                repository_id, task_type, task_area, risk_level, required_skills, None, top_k * 2
            )
            
            if not basic_recommendations:
                logger.warning(f"No basic recommendations found for repository {repository_id}")
                return []
            
            # Lấy workload hiện tại
            workloads = self.get_member_workload(repository_id)
            logger.info(f"Retrieved workload for {len(workloads)} members")
            
            # Điều chỉnh score dựa trên workload
            for rec in basic_recommendations:
                member = rec['member']
                workload = workloads.get(member, {
                    'workload_score': 0.0,
                    'active_tasks': 0,
                    'in_progress_tasks': 0,
                    'high_priority_tasks': 0
                })
                
                # Penalty cho workload cao
                workload_penalty = min(20, workload.get('workload_score', 0) * 2)  # Max 20 points penalty
                
                # Bonus cho members có ít việc
                if workload.get('workload_score', 0) < 2:
                    workload_bonus = 5
                else:
                    workload_bonus = 0
                
                # Điều chỉnh score
                original_score = rec['score']
                adjusted_score = original_score - workload_penalty + workload_bonus
                rec['adjusted_score'] = max(0, adjusted_score)
                
                # Cập nhật explanation
                active_tasks = workload.get('active_tasks', 0)
                if active_tasks > 0:
                    rec['explanation'] += f" (Hiện có {active_tasks} tasks active)"
                else:
                    rec['explanation'] += " (Hiện tại rảnh)"
                
                # Thêm workload info
                rec['workload_info'] = workload
            
            # Sắp xếp lại theo adjusted score
            basic_recommendations.sort(key=lambda x: x.get('adjusted_score', x['score']), reverse=True)
            
            return basic_recommendations[:top_k]
            
        except Exception as e:
            logger.error(f"Error in recommend_with_workload_balance: {str(e)}")
            # Fallback to basic recommendations without workload balancing
            try:
                return self.recommend_assignees(
                    repository_id, task_type, task_area, risk_level, required_skills, None, top_k
                )
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                return []
