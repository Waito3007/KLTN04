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
from services.multifusion_v2_service import MultiFusionV2Service
from services.area_analysis_service import AreaAnalysisService
from services.risk_analysis_service import RiskAnalysisService

logger = logging.getLogger(__name__)

class AssignmentRecommendationService:
    def __init__(self, db: Session):
        self.db = db
        # Initialize AI services
        self.multifusion_v2_service = MultiFusionV2Service()
        self.area_analysis_service = AreaAnalysisService()
        self.risk_analysis_service = RiskAnalysisService()
    
    def analyze_member_skills(self, repository_id: int, days_back: int = 90) -> Dict[str, Dict[str, Any]]:
        """
        Phân tích kỹ năng và thế mạnh của từng thành viên dựa trên commit history
        Sử dụng AI models: MultiFusion V2, Area Analyst, Risk Analyst
        
        Args:
            repository_id: ID của repository
            days_back: Số ngày quay lại để phân tích (mặc định 90 ngày)
            
        Returns:
            Dict với key là member login, value là skill profile với AI analysis
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Query để lấy thống kê commit của từng member
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
        
        member_skills = defaultdict(lambda: {
            'commit_types': defaultdict(int),
            'areas': defaultdict(int),
            'risk_levels': defaultdict(int),
            'languages': defaultdict(int),
            'total_commits': 0,
            'total_changes': 0,
            'avg_files_per_commit': 0.0,
            'recent_activity_score': 0.0,
            'expertise_areas': [],
            'risk_tolerance': 'low',
            'ai_analysis_count': 0,
            'ai_predictions': {
                'commit_types': defaultdict(int),
                'areas': defaultdict(int), 
                'risks': defaultdict(int)
            }
        })
        
        logger.info(f"Analyzing {len(commits_data)} commits with AI models...")
        
        for commit in commits_data:
            author = commit[0]
            message = commit[1]
            insertions = commit[2] or 0
            deletions = commit[3] or 0
            files_changed = commit[4] or 0
            modified_files = commit[5]
            commit_date = commit[6]
            diff_content = commit[8] or ''
            
            # Legacy analysis (fallback)
            commit_type_legacy = self._analyze_commit_type_from_message(message)
            area_legacy = self._analyze_area_from_files(modified_files)
            risk_level_legacy = self._analyze_risk_level(insertions, deletions, files_changed)
            language = self._detect_language_from_files(modified_files)
            
            # AI Analysis using the 3 models
            try:
                # 1. MultiFusion V2 for commit type
                ai_commit_prediction = self.multifusion_v2_service.predict_commit_type(
                    commit_message=message,
                    lines_added=insertions,
                    lines_removed=deletions,
                    files_count=files_changed,
                    detected_language=language
                )
                
                # 2. Area Analysis
                commit_data_for_area = {
                    'commit_message': message,
                    'diff_content': diff_content,
                    'files_count': files_changed,
                    'lines_added': insertions,
                    'lines_removed': deletions,
                    'total_changes': insertions + deletions
                }
                ai_area = self.area_analysis_service.predict_area(commit_data_for_area)
                
                # 3. Risk Analysis
                ai_risk = self.risk_analysis_service.predict_risk(commit_data_for_area)
                
                # Use AI predictions if available, fallback to legacy
                commit_type = ai_commit_prediction.get('commit_type', commit_type_legacy)
                area = ai_area if ai_area != "Model not loaded" else area_legacy
                risk_level = ai_risk if ai_risk != "Model not loaded" else risk_level_legacy
                
                # Track AI predictions
                profile = member_skills[author]
                profile['ai_predictions']['commit_types'][commit_type] += 1
                profile['ai_predictions']['areas'][area] += 1
                profile['ai_predictions']['risks'][risk_level] += 1
                profile['ai_analysis_count'] += 1
                
                logger.debug(f"AI Analysis for {author}: type={commit_type}, area={area}, risk={risk_level}")
                
            except Exception as e:
                logger.warning(f"AI analysis failed for commit by {author}, using fallback: {e}")
                commit_type = commit_type_legacy
                area = area_legacy
                risk_level = risk_level_legacy
            
            # Cập nhật skill profile
            profile = member_skills[author]
            profile['commit_types'][commit_type] += 1
            profile['areas'][area] += 1
            profile['risk_levels'][risk_level] += 1
            profile['languages'][language] += 1
            profile['total_commits'] += 1
            profile['total_changes'] += insertions + deletions
            
            # Tính recent activity score (commits gần đây có trọng số cao hơn)
            days_ago = (datetime.now() - commit_date).days if commit_date else 90
            recency_weight = max(0.1, 1.0 - (days_ago / days_back))
            profile['recent_activity_score'] += recency_weight
        
        # Tính toán các metric cuối cùng
        for author, profile in member_skills.items():
            if profile['total_commits'] > 0:
                # Tính average files per commit
                total_files = sum(profile['areas'].values())
                profile['avg_files_per_commit'] = total_files / profile['total_commits']
                
                # Xác định expertise areas (top 2 areas, ưu tiên AI predictions)
                if profile['ai_analysis_count'] > 0:
                    # Use AI predictions for expertise areas
                    ai_areas = profile['ai_predictions']['areas']
                    sorted_areas = sorted(ai_areas.items(), key=lambda x: x[1], reverse=True)
                else:
                    # Fallback to legacy analysis
                    sorted_areas = sorted(profile['areas'].items(), key=lambda x: x[1], reverse=True)
                
                profile['expertise_areas'] = [area for area, count in sorted_areas[:2] if count >= 2]
                
                # Xác định risk tolerance (ưu tiên AI predictions)
                if profile['ai_analysis_count'] > 0:
                    ai_risks = profile['ai_predictions']['risks']
                    total_ai_commits = sum(ai_risks.values())
                    high_risk_ratio = (ai_risks.get('highrisk', 0) + ai_risks.get('high', 0)) / total_ai_commits if total_ai_commits > 0 else 0
                else:
                    # Fallback to legacy analysis
                    high_risk_ratio = profile['risk_levels']['high'] / profile['total_commits']
                
                if high_risk_ratio > 0.3:
                    profile['risk_tolerance'] = 'high'
                elif high_risk_ratio > 0.1:
                    profile['risk_tolerance'] = 'medium'
                else:
                    profile['risk_tolerance'] = 'low'
                
                # Add AI analysis summary
                profile['ai_coverage'] = profile['ai_analysis_count'] / profile['total_commits'] if profile['total_commits'] > 0 else 0
        
        logger.info(f"Completed AI-enhanced analysis for {len(member_skills)} members")
        return dict(member_skills)
    
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
        member_skills = self.analyze_member_skills(repository_id)
        
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
        
        if use_ai and 'ai_predictions' in profile:
            # Use AI predictions for scoring
            ai_commit_types = profile['ai_predictions']['commit_types']
            ai_areas = profile['ai_predictions']['areas']
            ai_risks = profile['ai_predictions']['risks']
            
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
            # Fallback to legacy analysis
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
        
        # 5. Overall experience (10% of total score)
        total_commits = profile['total_commits']
        experience_score = min(1.0, total_commits / 50)  # Normalize với max 50 commits
        score += experience_score * 10
        
        # Bonus for required skills
        if required_skills:
            skill_bonus = 0
            for skill in required_skills:
                if skill in profile['languages'] and profile['languages'][skill] > 0:
                    skill_bonus += 5
            score += min(skill_bonus, 15)  # Max 15 bonus points
        
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
        
        if use_ai and 'ai_predictions' in profile:
            # AI-enhanced explanations
            ai_commit_types = profile['ai_predictions']['commit_types']
            ai_areas = profile['ai_predictions']['areas']
            ai_risks = profile['ai_predictions']['risks']
            
            # Experience với task type (AI)
            task_type_count = ai_commit_types.get(task_type, 0)
            if task_type_count > 0:
                explanations.append(f"AI phân tích: {task_type_count} commits loại {task_type}")
            
            # Experience với area (AI)
            area_count = ai_areas.get(task_area, 0)
            if area_count > 0:
                explanations.append(f"AI phân tích: {area_count} commits trong {task_area}")
            
            # Risk tolerance (AI)
            ai_risk_tolerance = self._calculate_ai_risk_tolerance(ai_risks)
            if ai_risk_tolerance == 'high' and risk_level == 'high':
                explanations.append("AI xác nhận: Chuyên xử lý tasks rủi ro cao")
            elif ai_risk_tolerance == 'low' and risk_level == 'low':
                explanations.append("AI xác nhận: Chuyên xử lý tasks ổn định")
                
            # AI coverage info
            ai_coverage_pct = int(profile['ai_coverage'] * 100)
            explanations.append(f"Phân tích AI: {ai_coverage_pct}% commits")
            
        else:
            # Legacy explanations
            # Experience với task type
            task_type_count = profile['commit_types'].get(task_type, 0)
            if task_type_count > 0:
                explanations.append(f"Có {task_type_count} commits loại {task_type}")
            
            # Experience với area
            area_count = profile['areas'].get(task_area, 0)
            if area_count > 0:
                explanations.append(f"Có {area_count} commits trong {task_area}")
            
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
            explanations.append(f"Chuyên gia về {task_area}")
        
        return "; ".join(explanations) if explanations else "Thành viên phù hợp dựa trên tổng thể"
    
    def _analyze_commit_type_from_message(self, message: str) -> str:
        """Phân tích loại commit từ message"""
        if not message:
            return 'other'
        
        message_lower = message.lower()
        
        # Conventional commit format
        if message_lower.startswith('feat'):
            return 'feat'
        elif message_lower.startswith('fix'):
            return 'fix'
        elif message_lower.startswith('docs'):
            return 'docs'
        elif message_lower.startswith('refactor'):
            return 'refactor'
        elif message_lower.startswith('test'):
            return 'test'
        elif message_lower.startswith('chore'):
            return 'chore'
        elif message_lower.startswith('style'):
            return 'style'
        elif message_lower.startswith('perf'):
            return 'perf'
        
        # Keyword-based detection
        if any(word in message_lower for word in ['feature', 'add', 'implement']):
            return 'feat'
        elif any(word in message_lower for word in ['fix', 'bug', 'error', 'issue']):
            return 'fix'
        elif any(word in message_lower for word in ['doc', 'readme', 'comment']):
            return 'docs'
        elif any(word in message_lower for word in ['refactor', 'cleanup', 'restructure']):
            return 'refactor'
        elif any(word in message_lower for word in ['test', 'spec', 'coverage']):
            return 'test'
        
        return 'other'
    
    def _analyze_area_from_files(self, modified_files) -> str:
        """Phân tích area từ danh sách files được modify"""
        if not modified_files:
            return 'general'
        
        try:
            import json
            if isinstance(modified_files, str):
                files_list = json.loads(modified_files)
            elif isinstance(modified_files, list):
                files_list = modified_files
            else:
                return 'general'
        except:
            return 'general'
        
        area_indicators = {
            'frontend': ['.jsx', '.tsx', '.js', '.ts', '.vue', '.css', '.scss', '.html'],
            'backend': ['.py', '.java', '.go', '.php', '.rb', '.cs'],
            'database': ['.sql', 'migration', 'schema'],
            'devops': ['dockerfile', '.yml', '.yaml', '.sh', '.json', 'docker'],
            'mobile': ['.swift', '.kt', '.dart'],
            'docs': ['.md', '.txt', '.rst', 'readme']
        }
        
        area_scores = defaultdict(int)
        
        for file_path in files_list:
            if isinstance(file_path, str):
                file_lower = file_path.lower()
                
                for area, indicators in area_indicators.items():
                    for indicator in indicators:
                        if indicator in file_lower:
                            area_scores[area] += 1
        
        if area_scores:
            return max(area_scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def _analyze_risk_level(self, insertions: int, deletions: int, files_changed: int) -> str:
        """Phân tích mức độ rủi ro dựa trên metrics"""
        total_changes = insertions + deletions
        
        # High risk indicators
        if (total_changes > 1000 or 
            files_changed > 20 or 
            deletions > 500):
            return 'high'
        
        # Medium risk indicators  
        if (total_changes > 200 or 
            files_changed > 5 or
            deletions > 100):
            return 'medium'
        
        return 'low'
    
    def _detect_language_from_files(self, modified_files) -> str:
        """Phát hiện ngôn ngữ lập trình chính"""
        if not modified_files:
            return 'unknown'
        
        try:
            import json
            if isinstance(modified_files, str):
                files_list = json.loads(modified_files)
            elif isinstance(modified_files, list):
                files_list = modified_files
            else:
                return 'unknown'
        except:
            return 'unknown'
        
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript', 
            '.jsx': 'JavaScript',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C++',
            '.cs': 'C#',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.go': 'Go',
            '.rs': 'Rust',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'CSS'
        }
        
        language_count = defaultdict(int)
        for file_path in files_list:
            if isinstance(file_path, str):
                import os
                _, ext = os.path.splitext(file_path.lower())
                language = language_map.get(ext, 'other')
                language_count[language] += 1
        
        if language_count:
            return max(language_count.items(), key=lambda x: x[1])[0]
        
        return 'unknown'

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
