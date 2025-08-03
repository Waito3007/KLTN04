from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta
from collections import defaultdict
import re
import asyncio
import logging

# Set up logger
logger = logging.getLogger(__name__)

class MemberAnalysisService:
    def __init__(self, db: Session):        
        self.db = db
    
    def get_repository_members(self, repository_id: int) -> List[Dict[str, Any]]:
        """Lấy danh sách members của repository trực tiếp từ commit authors - KHÔNG MAPPING"""
        # Get all unique commit authors directly from commits table
        query = text("""
            SELECT 
                author_name,
                COUNT(*) as total_commits
            FROM commits 
            WHERE repo_id = :repo_id
            GROUP BY author_name
            HAVING COUNT(*) > 0
            ORDER BY total_commits DESC
        """)
        
        result = self.db.execute(query, {"repo_id": repository_id}).fetchall()
        
        members = []
        for row in result:
            author_name = row[0]
            commit_count = row[1]
            
            members.append({
                "id": f"author_{author_name}",  # Use author name as ID
                "login": author_name,
                "display_name": author_name,
                "avatar_url": None,  # No avatar for commit authors
                "total_commits": commit_count
            })
        
        return members
          # Get commit authors who aren't formal collaborators
        unmatched_query = text("""
            SELECT 
                author_name,
                author_email,
                COUNT(*) as total_commits
            FROM commits 
            WHERE repo_id = :repo_id
            GROUP BY author_name, author_email
            HAVING COUNT(*) > 0
            ORDER BY total_commits DESC
        """)
        
        unmatched_result = self.db.execute(unmatched_query, {"repo_id": repository_id}).fetchall()
        
        # Group authors by email first (same email = same person)
        email_to_authors = {}
        for row in unmatched_result:
            author_name = row[0]
            author_email = row[1]
            commit_count = row[2]
            
            if author_email not in email_to_authors:
                email_to_authors[author_email] = []
            email_to_authors[author_email].append((author_name, commit_count))
          # Process each email group
        for author_email, authors_data in email_to_authors.items():
            # Consolidate authors with same email
            total_commits_for_email = sum(count for _, count in authors_data)
            # Use the author name with most commits as primary
            primary_author = max(authors_data, key=lambda x: x[1])[0]
              # CONSERVATIVE: Only check for exact email matches with GitHub noreply
            email_matches_collaborator = False
            collaborator_member = None
            
            # Only merge if it's a GitHub noreply email with EXACT username match
            if author_email and '@users.noreply.github.com' in author_email:
                # Extract username from GitHub noreply email
                email_parts = author_email.split('@')[0]
                if '+' in email_parts:
                    github_username = email_parts.split('+')[1]
                    # Only merge if there's an EXACT collaborator with this username
                    for member in members:
                        if (member['id'] != f"author_{member['login']}" and  # This is a formal collaborator
                            github_username.lower() == member['login'].lower()):
                            email_matches_collaborator = True
                            collaborator_member = member
                            break
            
            if email_matches_collaborator and collaborator_member:
                # Merge with existing collaborator (add commit counts)
                collaborator_member['total_commits'] += total_commits_for_email
                # Mark all author names as processed
                for author_name, _ in authors_data:
                    processed_authors.add(normalize_name(author_name))
                continue
              # Check if this email/author is already covered by a collaborator (name-based)
            is_already_processed = False
            for processed_name in processed_authors:
                for author_name, _ in authors_data:
                    if are_names_similar(author_name, processed_name):
                        is_already_processed = True
                        break
                if is_already_processed:
                    break
              # CONSERVATIVE: Only check for exact name matches with collaborators
            if not is_already_processed:
                collaborator_match = None
                for member in members:
                    if member['id'] != f"author_{member['login']}":  # This is a formal collaborator
                        for author_name, _ in authors_data:
                            # Only exact matches (case-insensitive)
                            if (normalize_name(author_name) == normalize_name(member['login']) or 
                                normalize_name(author_name) == normalize_name(member['display_name'] or '')):
                                collaborator_match = member
                                break
                    if collaborator_match:
                        break
                
                if collaborator_match:
                    # Merge with existing collaborator (add commit counts)
                    collaborator_match['total_commits'] += total_commits_for_email
                    # Mark all author names as processed
                    for author_name, _ in authors_data:
                        processed_authors.add(normalize_name(author_name))
                    continue
            
            if not is_already_processed:
                # Check if we already have a similar author in our members list (name-based)
                similar_member = None
                for member in members:
                    for author_name, _ in authors_data:
                        if (are_names_similar(author_name, member['login']) or 
                            are_names_similar(author_name, member['display_name'])):
                            similar_member = member
                            break
                    if similar_member:
                        break
                
                if similar_member:
                    # Merge with existing member (add commit counts)
                    similar_member['total_commits'] += total_commits_for_email
                    # Use the name with more commits as primary
                    if total_commits_for_email > similar_member['total_commits'] - total_commits_for_email:
                        similar_member['login'] = primary_author
                        similar_member['display_name'] = primary_author
                else:
                    # Add as new informal member (consolidated by email)
                    members.append({
                        "id": f"author_{primary_author}",  # Special ID for non-collaborator authors
                        "login": primary_author,
                        "display_name": primary_author,
                        "avatar_url": None,
                        "total_commits": total_commits_for_email
                    })
                
                # Mark all author names as processed
                for author_name, _ in authors_data:
                    processed_authors.add(normalize_name(author_name))
        
        # Sort by commit count
        members.sort(key=lambda x: x['total_commits'], reverse=True)
        
        return members
    
    def get_repository_branches(self, repository_id: int) -> List[Dict[str, Any]]:
        """Lấy danh sách branches của repository"""
        query = text("""
            SELECT 
                b.id,
                b.name,
                b.is_default,
                b.commits_count,
                b.last_commit_date,
                COUNT(c.id) as actual_commits_count
            FROM branches b
            LEFT JOIN commits c ON c.branch_name = b.name AND c.repo_id = :repo_id
            WHERE b.repo_id = :repo_id
            GROUP BY b.id, b.name, b.is_default, b.commits_count, b.last_commit_date
            ORDER BY b.is_default DESC, b.commits_count DESC, b.name ASC
        """)
        
        result = self.db.execute(query, {"repo_id": repository_id}).fetchall()
        
        branches = []
        for row in result:
            branches.append({
                "id": row[0],
                "name": row[1],
                "is_default": row[2] or False,
                "commits_count": row[5],  # actual count from commits table
                "last_commit_date": row[4].isoformat() if row[4] else None
            })
        return branches
    # lấy commits của member với phân tích đơn giản và branch filter
    def get_member_commits_with_analysis(
        self, 
        repository_id: int, 
        member_login: str, 
        limit: int = 1000,
        branch_name: str = None  # NEW: Optional branch filter
    ) -> Dict[str, Any]:
        """Lấy commits của member với analysis đơn giản và branch filter"""
        
        # ENHANCED: Get all author names associated with this member
        all_author_names = self._get_all_author_names_for_member(repository_id, member_login)
        
        if not all_author_names:
            all_author_names = [member_login]  # Fallback
        
        # Create IN clause for multiple author names
        author_placeholders = ', '.join([f':author_{i}' for i in range(len(all_author_names))])
        
        # Query commits của member với multiple author names và branch filter
        base_query = f"""
            SELECT 
                id, sha, message, author_name, author_email,
                date, branch_name, insertions, deletions, files_changed, modified_files
            FROM commits 
            WHERE repo_id = :repo_id 
                AND LOWER(author_name) IN ({author_placeholders})
        """
        
        params = {
            "repo_id": repository_id,
            "limit": limit
        }
        
        # thêm tên tác giả vào params
        for i, author_name in enumerate(all_author_names):
            params[f"author_{i}"] = author_name.lower()
        
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
                    "ai_powered": False,
                    "analysis_date": datetime.now().isoformat()
                },
                "commits": [],
                "statistics": {"commit_types": {}, "tech_analysis": {}, "productivity": {"total_additions": 0, "total_deletions": 0}}
            }
          # phân tích commits
        commits_with_analysis = []
        commit_type_stats = defaultdict(int)
        tech_stats = defaultdict(int)
        total_additions = 0
        total_deletions = 0
        
        for commit in commits_data:
            # Simple pattern-based analysis
            analysis = self._analyze_commit_message(commit[2])  # commit.message
            
            # Detect language from modified files
            detected_language = self._detect_language_from_files(commit[10])  # modified_files is index 10
            
            commit_info = {
                "id": commit[0],
                "sha": commit[1][:8] if commit[1] else "N/A",
                "message": commit[2],
                "author": commit[3],
                "date": commit[5].isoformat() if commit[5] else None,  # date is index 5
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
                    "type": analysis["type"],
                    "type_icon": analysis["icon"],
                    "tech_area": analysis["tech_area"],
                    "ai_powered": False
                }
            }
            
            commits_with_analysis.append(commit_info)
            commit_type_stats[analysis["type"]] += 1
            tech_stats[analysis["tech_area"]] += 1
            total_additions += commit[7] or 0
            total_deletions += commit[8] or 0
        
        return {
            "member": {"login": member_login, "display_name": member_login},
            "summary": {
                "total_commits": len(commits_with_analysis),
                "branch_filter": branch_name,
                "ai_powered": False,
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
            }        }

    def _analyze_commit_message(self, message: str) -> Dict[str, str]:
        """Simple pattern-based commit analysis"""
        message_lower = message.lower()
        
        # Determine type
        if any(word in message_lower for word in ['feat', 'feature', 'add', 'implement']):
            commit_type = "feat"
            icon = "🚀"
        elif any(word in message_lower for word in ['fix', 'bug', 'error', 'issue']):
            commit_type = "fix"
            icon = "🐛"
        elif any(word in message_lower for word in ['docs', 'documentation', 'readme']):
            commit_type = "docs"
            icon = "📝"
        elif any(word in message_lower for word in ['chore', 'cleanup', 'refactor']):
            commit_type = "chore"
            icon = "🔧"
        else:
            commit_type = "other"
            icon = "📦"
        
        # Determine tech area
        if any(word in message_lower for word in ['api', 'endpoint', 'rest', 'graphql']):
            tech_area = "API"
        elif any(word in message_lower for word in ['ui', 'frontend', 'react', 'vue', 'component']):
            tech_area = "Frontend"
        elif any(word in message_lower for word in ['database', 'db', 'sql', 'migration']):
            tech_area = "Database"
        elif any(word in message_lower for word in ['test', 'testing', 'spec', 'unittest']):
            tech_area = "Testing"
        else:
            tech_area = "General"
        
        return {
            "type": commit_type,
            "icon": icon,
            "tech_area": tech_area
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
            "other": "📦"        }
        return icons.get(commit_type, "📦")
    
    def _get_all_author_names_for_member(self, repository_id: int, member_login: str) -> List[str]:
        """Get exact author name for member - NO MAPPING, NO CONSOLIDATION"""
        # Return only the exact member_login as author name
        # Each author_name in commits should be treated as separate contributor
        return [member_login]

    def _are_names_similar(self, name1: str, name2: str) -> bool:
        """Basic name similarity check."""
        if not name1 or not name2:
            return False
        return name1.lower() in name2.lower() or name2.lower() in name1.lower()
    
    def _get_member_commits_raw(self, repository_id: int, member_login: str, limit: int = 1000, branch_name: str = None):
        """Get raw commit data for a member - EXACT CASE-SENSITIVE MATCH"""
        
        # No author name mapping - use exact member_login only
        author_name = member_login
        
        # Query commits with EXACT case-sensitive author name match
        base_query = """
            SELECT 
                sha, author_name, message, committer_date, 
                insertions, deletions, files_changed, modified_files, diff_content
            FROM commits 
            WHERE repo_id = :repo_id 
                AND author_name = :author_name
        """
        
        params = {
            "repo_id": repository_id,
            "author_name": author_name,
            "limit": limit
        }
        
        # Add branch filter if specified
        if branch_name:
            base_query += " AND branch_name = :branch_name"
            params["branch_name"] = branch_name
            
        base_query += " ORDER BY committer_date DESC LIMIT :limit"
        
        query = text(base_query)
        
        commits_data = self.db.execute(query, params).fetchall()
        
        # Add detected language to each commit
        enhanced_commits = []
        for commit in commits_data:
            detected_language = self._detect_language_from_files(commit[7])  # modified_files is index 7
            enhanced_commit = list(commit) + [detected_language]  # Add detected_language
            enhanced_commits.append(enhanced_commit)
        
        return enhanced_commits
    
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

    