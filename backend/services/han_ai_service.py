# backend/services/han_ai_service.py
"""
HAN AI Service - Service layer for HAN model integration
Provides high-level API for commit analysis and project management AI features
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio
from functools import lru_cache

# Add AI directory to path
ai_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ai')
sys.path.insert(0, ai_dir)

try:
    from ai.han_commit_analyzer import HANCommitAnalyzer
except ImportError as e:
    logging.warning(f"HAN model not available: {e}")
    HANCommitAnalyzer = None

logger = logging.getLogger(__name__)

class HANAIService:
    """
    Service class for HAN-based AI analysis in project management
    """
    
    def __init__(self):
        self.analyzer = None
        self.is_model_loaded = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize HAN model with error handling"""
        try:
            if HANCommitAnalyzer is None:
                logger.warning("HAN analyzer not available - using mock responses")
                return
                
            self.analyzer = HANCommitAnalyzer()
            self.analyzer.load_model()
            self.is_model_loaded = True
            logger.info("HAN model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load HAN model: {e}")
            self.is_model_loaded = False
    
    async def analyze_commit_message(self, message: str) -> Dict[str, Any]:
        """
        Analyze a single commit message
        
        Args:
            message: Commit message text
            
        Returns:
            Analysis results including category, impact, urgency
        """
        try:
            if not self.is_model_loaded:
                return self._mock_commit_analysis(message)
            
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.analyzer.predict_commit_analysis, 
                message
            )
            
            return {
                'success': True,
                'message': message,
                'analysis': result,
                'model_version': 'HAN-v1',
                'confidence': result.get('confidence', {})
            }
            
        except Exception as e:
            logger.error(f"Error analyzing commit: {e}")
            return {
                'success': False,
                'message': message,
                'error': str(e),
                'analysis': self._mock_commit_analysis(message)['analysis']
            }
    
    async def analyze_commits_batch(self, messages: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple commit messages in batch
        
        Args:
            messages: List of commit message texts
            
        Returns:
            Batch analysis results
        """
        results = []
        
        for message in messages:
            analysis = await self.analyze_commit_message(message)
            results.append(analysis)
        
        # Generate batch statistics
        stats = self._calculate_batch_statistics(results)
        
        return {
            'success': True,
            'total_commits': len(messages),
            'results': results,
            'statistics': stats
        }
    
    # Alias for backward compatibility
    analyze_commits = analyze_commits_batch
    
    async def analyze_developer_patterns(self, developer_commits: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze commit patterns for each developer
        
        Args:
            developer_commits: Dict mapping developer name to list of commits
            
        Returns:
            Developer pattern analysis
        """
        developer_profiles = {}
        
        for developer, commits in developer_commits.items():
            if not commits:
                continue
                
            # Analyze all commits for this developer
            batch_result = await self.analyze_commits_batch(commits)
            
            # Create developer profile
            profile = self._create_developer_profile(commits, batch_result)
            developer_profiles[developer] = profile
        
        return {
            'success': True,
            'developer_profiles': developer_profiles,
            'total_developers': len(developer_profiles)
        }
    
    async def suggest_task_assignment(self, tasks: List[Dict], developers: List[Dict]) -> Dict[str, Any]:
        """
        Suggest task assignments based on commit analysis and developer profiles
        
        Args:
            tasks: List of task dictionaries
            developers: List of developer dictionaries with commit history
            
        Returns:
            Task assignment suggestions
        """
        try:
            # Analyze developer patterns first
            developer_commits = {}
            for dev in developers:
                developer_commits[dev['login']] = dev.get('recent_commits', [])
            
            developer_analysis = await self.analyze_developer_patterns(developer_commits)
            
            # Generate task assignments
            assignments = []
            for task in tasks:
                assignment = self._match_task_to_developer(
                    task, 
                    developer_analysis['developer_profiles']
                )
                assignments.append(assignment)
            
            return {
                'success': True,
                'assignments': assignments,
                'developer_analysis': developer_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in task assignment: {e}")
            return {
                'success': False,
                'error': str(e),
                'assignments': self._mock_task_assignments(tasks, developers)
            }
    
    async def generate_project_insights(self, project_data: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive project insights based on commit analysis
        
        Args:
            project_data: Project data including commits, contributors, etc.
            
        Returns:
            Project insights and recommendations
        """
        try:
            all_commits = project_data.get('commits', [])
            contributors = project_data.get('contributors', [])
            
            # Analyze all commits
            commit_messages = [commit.get('message', '') for commit in all_commits]
            batch_analysis = await self.analyze_commits_batch(commit_messages)
            
            # Generate insights
            insights = {
                'commit_analysis': batch_analysis,
                'code_quality_trends': self._analyze_quality_trends(batch_analysis),
                'team_collaboration': self._analyze_team_collaboration(all_commits, contributors),
                'project_health': self._assess_project_health(batch_analysis),
                'recommendations': self._generate_recommendations(batch_analysis)
            }
            
            return {
                'success': True,
                'project_name': project_data.get('name', 'Unknown'),
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error generating project insights: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _mock_commit_analysis(self, message: str) -> Dict[str, Any]:
        """Mock analysis when model is not available"""
        # Simple keyword-based mock analysis
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['fix', 'bug', 'error', 'issue']):
            category = 'bug'
            impact = 'medium'
            urgency = 'high'
        elif any(word in message_lower for word in ['feat', 'feature', 'add', 'new']):
            category = 'feature'
            impact = 'high'
            urgency = 'medium'
        elif any(word in message_lower for word in ['docs', 'doc', 'readme']):
            category = 'docs'
            impact = 'low'
            urgency = 'low'
        elif any(word in message_lower for word in ['test', 'spec']):
            category = 'test'
            impact = 'medium'
            urgency = 'low'
        else:
            category = 'chore'
            impact = 'low'
            urgency = 'medium'
        
        return {
            'success': True,
            'analysis': {
                'category': category,
                'impact': impact,
                'urgency': urgency,
                'is_mock': True
            }
        }
    
    def _calculate_batch_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics from batch analysis results"""
        if not results:
            return {}
        
        categories = {}
        impacts = {}
        urgencies = {}
        successful = 0
        
        for result in results:
            if result.get('success', False):
                successful += 1
                analysis = result.get('analysis', {})
                
                # Count categories
                category = analysis.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
                
                # Count impacts
                impact = analysis.get('impact', 'unknown')
                impacts[impact] = impacts.get(impact, 0) + 1
                
                # Count urgencies
                urgency = analysis.get('urgency', 'unknown')
                urgencies[urgency] = urgencies.get(urgency, 0) + 1
        
        return {
            'total_analyzed': len(results),
            'successful_analyses': successful,
            'success_rate': successful / len(results) if results else 0,
            'category_distribution': categories,
            'impact_distribution': impacts,
            'urgency_distribution': urgencies
        }
    
    def _create_developer_profile(self, commits: List[str], analysis_result: Dict) -> Dict[str, Any]:
        """Create developer profile from commit analysis"""
        stats = analysis_result.get('statistics', {})
        
        return {
            'total_commits': len(commits),
            'preferred_categories': self._get_top_categories(stats.get('category_distribution', {})),
            'impact_pattern': self._get_top_categories(stats.get('impact_distribution', {})),
            'urgency_pattern': self._get_top_categories(stats.get('urgency_distribution', {})),
            'activity_score': min(len(commits) / 10, 10),  # Scale 0-10
            'specialization': self._determine_specialization(stats)
        }
    
    def _get_top_categories(self, distribution: Dict[str, int], top_n: int = 3) -> List[str]:
        """Get top N categories from distribution"""
        if not distribution:
            return []
        
        sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:top_n]]
    
    def _determine_specialization(self, stats: Dict) -> str:
        """Determine developer specialization based on commit patterns"""
        categories = stats.get('category_distribution', {})
        
        if not categories:
            return 'generalist'
        
        top_category = max(categories.items(), key=lambda x: x[1])
        total_commits = sum(categories.values())
        
        if top_category[1] / total_commits > 0.5:
            return f"{top_category[0]}_specialist"
        else:
            return 'generalist'
    
    def _match_task_to_developer(self, task: Dict, developer_profiles: Dict) -> Dict[str, Any]:
        """Match a task to the best developer based on profiles"""
        task_type = task.get('type', 'feature').lower()
        task_priority = task.get('priority', 'medium').lower()
        
        best_match = None
        best_score = 0
        
        for dev_name, profile in developer_profiles.items():
            score = self._calculate_match_score(task_type, task_priority, profile)
            if score > best_score:
                best_score = score
                best_match = dev_name
        
        return {
            'task_title': task.get('title', 'Untitled'),
            'task_type': task_type,
            'recommended_developer': best_match or 'No suitable match',
            'confidence_score': best_score,
            'reasoning': self._generate_assignment_reasoning(task, best_match, developer_profiles.get(best_match, {}))
        }
    
    def _calculate_match_score(self, task_type: str, task_priority: str, profile: Dict) -> float:
        """Calculate match score between task and developer"""
        score = 0.0
        
        # Check category preference
        preferred_categories = profile.get('preferred_categories', [])
        if task_type in preferred_categories:
            score += 0.4
        
        # Check specialization
        specialization = profile.get('specialization', '')
        if task_type in specialization:
            score += 0.3
        
        # Activity score
        activity_score = profile.get('activity_score', 0)
        score += (activity_score / 10) * 0.3
        
        return score
    
    def _generate_assignment_reasoning(self, task: Dict, developer: str, profile: Dict) -> str:
        """Generate reasoning for task assignment"""
        if not developer or not profile:
            return "No suitable developer found based on commit analysis"
        
        specialization = profile.get('specialization', 'generalist')
        activity_score = profile.get('activity_score', 0)
        
        return f"Recommended {developer} based on {specialization} specialization and activity score of {activity_score:.1f}/10"
    
    def _mock_task_assignments(self, tasks: List[Dict], developers: List[Dict]) -> List[Dict]:
        """Generate mock task assignments when model is unavailable"""
        assignments = []
        
        for i, task in enumerate(tasks):
            dev_index = i % len(developers) if developers else 0
            developer = developers[dev_index]['login'] if developers else 'Unknown'
            
            assignments.append({
                'task_title': task.get('title', 'Untitled'),
                'recommended_developer': developer,
                'confidence_score': 0.5,
                'reasoning': 'Mock assignment - HAN model not available'
            })
        
        return assignments
    
    def _analyze_quality_trends(self, analysis: Dict) -> Dict[str, Any]:
        """Analyze code quality trends from commit analysis"""
        stats = analysis.get('statistics', {})
        categories = stats.get('category_distribution', {})
        
        bug_ratio = categories.get('bug', 0) / max(stats.get('total_analyzed', 1), 1)
        test_ratio = categories.get('test', 0) / max(stats.get('total_analyzed', 1), 1)
        
        quality_score = max(0, 1 - bug_ratio + test_ratio * 0.5)
        
        return {
            'quality_score': round(quality_score, 2),
            'bug_fix_ratio': round(bug_ratio, 2),
            'test_coverage_indicator': round(test_ratio, 2),
            'trend': 'improving' if quality_score > 0.7 else 'needs_attention'
        }
    
    def _analyze_team_collaboration(self, commits: List[Dict], contributors: List[Dict]) -> Dict[str, Any]:
        """Analyze team collaboration patterns"""
        return {
            'total_contributors': len(contributors),
            'commit_distribution': 'balanced',  # Simplified
            'collaboration_score': 0.8  # Mock score
        }
    
    def _assess_project_health(self, analysis: Dict) -> Dict[str, Any]:
        """Assess overall project health"""
        stats = analysis.get('statistics', {})
        success_rate = stats.get('success_rate', 0)
        
        health_score = success_rate * 0.8 + 0.2  # Base score
        
        return {
            'health_score': round(health_score, 2),
            'status': 'healthy' if health_score > 0.7 else 'needs_attention',
            'analysis_coverage': f"{stats.get('successful_analyses', 0)}/{stats.get('total_analyzed', 0)}"
        }
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate project recommendations based on analysis"""
        recommendations = []
        stats = analysis.get('statistics', {})
        categories = stats.get('category_distribution', {})
        
        total = sum(categories.values()) if categories else 1
        
        if categories.get('bug', 0) / total > 0.3:
            recommendations.append("Consider increasing code review practices to reduce bug fixes")
        
        if categories.get('test', 0) / total < 0.1:
            recommendations.append("Increase test coverage to improve code quality")
        
        if categories.get('docs', 0) / total < 0.05:
            recommendations.append("Improve documentation practices")
        
        if not recommendations:
            recommendations.append("Project shows good development practices")
        
        return recommendations

# Singleton instance
_han_ai_service = None

def get_han_ai_service() -> HANAIService:
    """Get singleton instance of HAN AI Service"""
    global _han_ai_service
    if _han_ai_service is None:
        _han_ai_service = HANAIService()
    return _han_ai_service
