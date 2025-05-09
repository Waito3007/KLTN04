# KLTN04\backend\services\commit_service.py
from .model_loader import predict_commit
from typing import List, Dict

class CommitService:
    @staticmethod
    def analyze_commits(commits: List[Dict]) -> Dict:
        """Phân tích danh sách commit"""
        results = {
            'total': len(commits),
            'critical': 0,
            'details': []
        }
        
        for commit in commits:
            is_critical = predict_commit(commit['message'])
            if is_critical:
                results['critical'] += 1
            results['details'].append({
                'id': commit['id'],
                'is_critical': is_critical,
                'message': commit['message'][:100] + '...' if len(commit['message']) > 100 else commit['message']
            })
        
        return results