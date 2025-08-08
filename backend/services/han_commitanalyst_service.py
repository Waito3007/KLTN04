from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta
from collections import defaultdict
import re
import asyncio
import logging
from services.han_ai_service import HANAIService

# Set up logger
logger = logging.getLogger(__name__)

class HanCommitAnalystService:
    def __init__(self, db: Session, ai_service: HANAIService):        
        self.db = db
        self.ai_service = ai_service

    async def get_member_commits_with_ai_analysis(
        self, 
        repository_id: int, 
        member_login: str, 
        limit: int = 1000,
        branch_name: str = None  # NEW: Optional branch filter
    ) -> Dict[str, Any]:
        """Lấy commits của member với AI analysis và branch filter"""
        
        # Get all author names associated with this member (including merged names)
        all_author_names = self._get_all_author_names_for_member(repository_id, member_login)
        
        # Build query to match any of the associated author names
        author_conditions = " OR ".join([f"LOWER(author_name) = LOWER(:author_name_{i})" for i in range(len(all_author_names))])
        
        base_query = f"""
            SELECT 
                id, sha, message, author_name, author_email,
                committer_date, branch_name, insertions, deletions, files_changed, modified_files
            FROM commits 
            WHERE repo_id = :repo_id 
                AND ({author_conditions})
        """
        
        params = {
            "repo_id": repository_id,
            "limit": limit
        }
        
        # Add all author names as parameters
        for i, author_name in enumerate(all_author_names):
            params[f"author_name_{i}"] = author_name
        
        # Add branch filter if specified
        if branch_name:
            base_query += " AND branch_name = :branch_name"
            params["branch_name"] = branch_name
            
        base_query += " ORDER BY committer_date DESC LIMIT :limit"
        
        query = text(base_query)
        
        commits_data = self.db.execute(query, params).fetchall()
        
        if not commits_data:
            return {
                "member": {"login": member_login, "display_name": member_login},
                "summary": {
                    "total_commits": 0, 
                    "message": f"No commits found{' on branch ' + branch_name if branch_name else ''}",
                    "branch_filter": branch_name,
                    "ai_powered": True,
                    "analysis_date": datetime.now().isoformat()
                },
                "commits": [],
                "statistics": {"commit_types": {}, "tech_analysis": {}, "productivity": {"total_additions": 0, "total_deletions": 0}}
            }
        
        # Prepare data for AI analysis
        commit_messages = [commit[2] for commit in commits_data]  # Extract messages
        
        # Get AI analysis
        try:
            # Sử dụng HAN AI Service mới
            ai_analysis_result = await self.ai_service.analyze_commits_batch(commit_messages)
            ai_analysis = {}
            # Chuẩn hóa kết quả: lấy từng commit analysis từ 'results' nếu có
            if ai_analysis_result and 'results' in ai_analysis_result:
                for idx, res in enumerate(ai_analysis_result['results']):
                    if res.get('success') and 'analysis' in res:
                        ai_analysis[idx] = res['analysis']
                    else:
                        ai_analysis[idx] = {}
            else:
                ai_analysis = {}
        except Exception as e:
            print(f"AI analysis failed, falling back to pattern analysis: {e}")
            # Fallback to pattern-based analysis
            return self.get_member_commits_with_analysis(repository_id, member_login, limit, branch_name)
        
        # Combine commits with AI analysis
        commits_with_analysis = []
        commit_type_stats = defaultdict(int)
        tech_stats = defaultdict(int)
        total_additions = 0
        total_deletions = 0
        
        for i, commit in enumerate(commits_data):
            ai_result = ai_analysis.get(i, {}) if ai_analysis else {}
            
            # Map lại các trường từ AI cho đúng chuẩn frontend
            # NEW: Ưu tiên lấy 'type' từ 'analysis', sau đó mới đến 'category'
            if 'analysis' in ai_result and 'type' in ai_result['analysis']:
                commit_type = ai_result['analysis']['type']
            else:
                commit_type = ai_result.get("type", ai_result.get("category", "other"))

            if 'analysis' in ai_result and 'tech_area' in ai_result['analysis']:
                tech_area = ai_result['analysis']['tech_area']
            else:
                tech_area = ai_result.get("tech_area", "general")
            
            # Detect language from modified files
            detected_language = self._detect_language_from_files(commit[10])  # modified_files is index 10
            
            commit_info = {
                "id": commit[0],
                "sha": commit[1][:8] if commit[1] else "N/A",
                "message": commit[2],
                "author": commit[3],
                "date": commit[5].isoformat() if commit[5] else None,
                "branch": commit[6] or "main",
                "insertions": commit[7] or 0,
                "deletions": commit[8] or 0,
                "files_changed": commit[9] or 0,
                "detected_language": detected_language,
                "stats": {
                    "insertions": commit[7] or 0,
                    "deletions": commit[8] or 0,
                    "files_changed": commit[9] or 0
                },
                "analysis": {
                    "type": commit_type,
                    "type_icon": self._get_type_icon(commit_type),
                    "tech_area": tech_area,
                    "impact": ai_result.get("impact", "medium"),
                    "urgency": ai_result.get("urgency", "normal"),
                    "ai_powered": True
                }
            }
            
            commits_with_analysis.append(commit_info)
            commit_type_stats[commit_type] += 1
            tech_stats[tech_area] += 1
            total_additions += commit[7] or 0
            total_deletions += commit[8] or 0
        
        # NEW: Return the raw AI analysis result directly
        return {
            "success": True,
            "member": {"login": member_login, "display_name": member_login},
            "summary": {
                "total_commits": len(commits_with_analysis),
                "branch_filter": branch_name,
                "ai_powered": True,
                "model_used": "HAN AI",
                "analysis_date": datetime.now().isoformat()
            },
            "commits": commits_with_analysis,
            "statistics": {
                "commit_types": dict(commit_type_stats),
                "tech_analysis": dict(tech_stats),
                "productivity": {
                    "total_additions": total_additions,
                    "total_deletions": total_deletions
                }
            },
            "raw_ai_analysis": ai_analysis_result # NEW: Expose raw analysis
        }

    def _get_type_icon(self, commit_type: str) -> str:
        """Get icon for commit type"""
        icons = {
            "feat": "🚀",
            "fix": "🐛", 
            "docs": "📝",
            "chore": "🔧",
            "refactor": "♻️",
            "test": "✅",
            "style": "💄",
            "other": "📦"
        }
        return icons.get(commit_type, "📦")

    def _get_all_author_names_for_member(self, repository_id: int, member_login: str) -> List[str]:
        """Get all author names associated with a member - MORE INCLUSIVE approach."""
        # Start with the primary login name
        author_names = {member_login.lower()}

        # Find all author names from the commits table for this repo
        query = text("""
            SELECT DISTINCT author_name, author_email
            FROM commits 
            WHERE repo_id = :repo_id
        """)
        all_authors = self.db.execute(query, {"repo_id": repository_id}).fetchall()

        # Find the primary email of the member from the collaborators table if they exist
        primary_email_query = text("""
            SELECT email FROM collaborators WHERE LOWER(github_username) = LOWER(:member_login)
        """)
        primary_email_result = self.db.execute(primary_email_query, {"member_login": member_login}).fetchone()
        primary_email = primary_email_result[0] if primary_email_result else None

        for author_name, author_email in all_authors:
            # 1. Match by primary email (if available)
            if primary_email and author_email and primary_email.lower() == author_email.lower():
                author_names.add(author_name.lower())
                continue

            # 2. Match by GitHub noreply email convention
            if author_email and '@users.noreply.github.com' in author_email:
                if f"+{member_login.lower()}@" in author_email.lower():
                    author_names.add(author_name.lower())
                    continue
            
            # 3. Match by name similarity (as a fallback)
            if self._are_names_similar(member_login, author_name):
                author_names.add(author_name.lower())

        return list(author_names)

    def _are_names_similar(self, name1: str, name2: str) -> bool:
        """Basic name similarity check."""
        if not name1 or not name2:
            return False
        return name1.lower() in name2.lower() or name2.lower() in name1.lower()

    def _detect_language_from_files(self, modified_files) -> str:
        """Phát hiện ngôn ngữ chính từ danh sách file thay đổi"""
        if not modified_files:
            return "unknown_language"
        
        # Parse JSON if it's a string
        import json
        try:
            if isinstance(modified_files, str):
                files_list = json.loads(modified_files)
            elif isinstance(modified_files, list):
                files_list = modified_files
            else:
                return "unknown_language"
        except:
            return "unknown_language"
        
        # Map file extensions to languages (phải khớp với metadata_v2.json)
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
            '.scss': 'CSS',
            '.sass': 'CSS',
            '.json': 'JSON',
            '.xml': 'XML',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.md': 'other',
            '.dockerfile': 'other'
        }
        
        # Count languages by file extensions
        language_count = {}
        for file_path in files_list:
            if isinstance(file_path, str):
                # Get file extension
                import os
                _, ext = os.path.splitext(file_path.lower())
                language = language_map.get(ext, 'other')
                language_count[language] = language_count.get(language, 0) + 1
        
        # Return most common language, fallback to Python
        if language_count:
            most_common_lang = max(language_count.items(), key=lambda x: x[1])[0]
            return most_common_lang if most_common_lang != 'other' else 'Python'
        
        return "Python"  # Default fallback
