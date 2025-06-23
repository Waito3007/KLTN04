from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta
from collections import defaultdict
import re
import asyncio
from services.han_ai_service import HANAIService

class MemberAnalysisService:
    def __init__(self, db: Session):        
        self.db = db
        self.ai_service = HANAIService()
    
    def get_repository_members(self, repository_id: int) -> List[Dict[str, Any]]:
        """Lấy danh sách members của repository từ collaborators table và commit authors"""
        # First get formal collaborators with their commit counts
        query = text("""
            SELECT 
                c.id,
                c.github_username,
                c.display_name,
                c.avatar_url,
                COUNT(co.id) as total_commits
            FROM collaborators c
            JOIN repository_collaborators rc ON c.id = rc.collaborator_id
            LEFT JOIN commits co ON (
                (LOWER(co.author_name) = LOWER(c.github_username)) OR 
                (LOWER(co.author_name) = LOWER(c.display_name))
            ) AND co.repo_id = :repo_id
            WHERE rc.repository_id = :repo_id
            GROUP BY c.id, c.github_username, c.display_name, c.avatar_url
            ORDER BY total_commits DESC
        """)
        
        result = self.db.execute(query, {"repo_id": repository_id}).fetchall()
        
        members = []
        processed_authors = set()
          # Helper function to normalize names for comparison
        def normalize_name(name):
            if not name:
                return ""
            # Remove accents and normalize to lowercase
            import unicodedata
            normalized = unicodedata.normalize('NFD', name)
            without_accents = ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')
            return without_accents.lower().strip()
        
        # Helper function to extract name parts for better matching
        def extract_name_parts(name):
            if not name:
                return set()
            # Split by common separators and extract meaningful parts
            parts = re.split(r'[\s\-_\.@]+', name.lower().strip())
            # Remove empty parts and very short parts (< 2 chars)
            meaningful_parts = {part for part in parts if len(part) >= 2}
            return meaningful_parts        # Helper function to check if names are similar
        def are_names_similar(name1, name2):
            if not name1 or not name2:
                return False
                
            n1, n2 = normalize_name(name1), normalize_name(name2)
            
            # Exact match
            if n1 == n2:
                return True
              # For very short names (like "San", "SAN"), be very strict
            # Only match if they are exact (case-insensitive) matches
            if len(n1) <= 3 or len(n2) <= 3:
                return n1 == n2
            
            # For longer names, check substring matches but be very careful
            # Require the shorter name to be at least 70% of the longer name
            if len(n1) > 4 and len(n2) > 4:
                shorter = min(n1, n2, key=len)
                longer = max(n1, n2, key=len)
                
                if shorter in longer:
                    # Check if the shorter name is substantial part of the longer name
                    ratio = len(shorter) / len(longer)
                    if ratio >= 0.7:  # At least 70% overlap
                        return True
            
            # Check name parts intersection (for complex names)
            # But be very strict about part matching
            parts1 = extract_name_parts(name1)
            parts2 = extract_name_parts(name2)
            
            if parts1 and parts2:
                # For single part names (like "San"), be very strict
                if len(parts1) == 1 and len(parts2) == 1:
                    # Only match if exactly the same
                    return parts1 == parts2
                
                # For multi-part names, require substantial overlap
                common_parts = parts1.intersection(parts2)
                if common_parts:
                    # Require multiple common parts OR one very long common part
                    if len(common_parts) >= 2:  # Multiple common parts
                        return True
                    elif len(common_parts) == 1:  # Single common part
                        part = list(common_parts)[0]
                        if len(part) > 4:  # Very long common part
                            part_ratio1 = len(part) / len(n1) if n1 else 0
                            part_ratio2 = len(part) / len(n2) if n2 else 0
                            if part_ratio1 > 0.7 or part_ratio2 > 0.7:
                                return True
            
            return False
        
        for row in result:
            github_username = row[1]
            display_name = row[2] or row[1]
            commit_count = row[4]
            
            members.append({
                "id": row[0],
                "login": github_username,  # Use github_username as primary login
                "display_name": display_name,
                "avatar_url": row[3],
                "total_commits": commit_count
            })
            
            # Track processed authors (case-insensitive and similar names)
            processed_authors.add(normalize_name(github_username))
            if display_name:
                processed_authors.add(normalize_name(display_name))
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
        limit: int = 50,
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
                date, branch_name, insertions, deletions, files_changed
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
            
            commit_info = {
                "id": commit[0],
                "sha": commit[1][:8] if commit[1] else "N/A",
                "message": commit[2],
                "author": commit[3],
                "date": commit[5].isoformat() if commit[5] else None,  # date is index 5
                "branch": commit[6] or "main",
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

    async def get_member_commits_with_ai_analysis(
        self, 
        repository_id: int, 
        member_login: str, 
        limit: int = 50,
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
                committer_date, branch_name, insertions, deletions, files_changed
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
            ai_analysis = await self.ai_service.analyze_commits(commit_messages)
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
            
            commit_info = {
                "id": commit[0],
                "sha": commit[1][:8] if commit[1] else "N/A",
                "message": commit[2],
                "author": commit[3],
                "date": commit[5].isoformat() if commit[5] else None,
                "branch": commit[6] or "main",
                "stats": {
                    "insertions": commit[7] or 0,
                    "deletions": commit[8] or 0,
                    "files_changed": commit[9] or 0
                },
                "analysis": {
                    "type": ai_result.get("type", "other"),
                    "type_icon": self._get_type_icon(ai_result.get("type", "other")),
                    "tech_area": ai_result.get("tech_area", "general"),
                    "impact": ai_result.get("impact", "medium"),
                    "urgency": ai_result.get("urgency", "normal"),
                    "ai_powered": True
                }
            }
            
            commits_with_analysis.append(commit_info)
            commit_type_stats[ai_result.get("type", "other")] += 1
            tech_stats[ai_result.get("tech_area", "general")] += 1
            total_additions += commit[7] or 0
            total_deletions += commit[8] or 0
        
        return {
            "member": {"login": member_login, "display_name": member_login},
            "summary": {
                "total_commits": len(commits_with_analysis),
                "branch_filter": branch_name,
                "ai_powered": True,
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
            }
        }
    
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
        """Get all author names associated with a member - CONSERVATIVE approach based on email evidence only"""
        author_names = []
        
        # 1. Add the member login itself
        author_names.append(member_login)
        
        # 2. Only add names that have CLEAR email evidence of being the same person
        query = text("""
            SELECT DISTINCT author_name, author_email
            FROM commits 
            WHERE repo_id = :repo_id
        """)
        
        all_authors = self.db.execute(query, {"repo_id": repository_id}).fetchall()
        
        for author_name, author_email in all_authors:
            # Only check GitHub noreply email pattern (most reliable)
            if author_email and '@users.noreply.github.com' in author_email:
                email_parts = author_email.split('@')[0]
                if '+' in email_parts:
                    github_username = email_parts.split('+')[1]
                    # Only match if GitHub username exactly matches member login
                    if github_username.lower() == member_login.lower():
                        if author_name not in author_names:
                            author_names.append(author_name)
            
            # Exact name match (case-insensitive) - very conservative
            if author_name.lower() == member_login.lower():
                if author_name not in author_names:
                    author_names.append(author_name)
        
        return author_names
    