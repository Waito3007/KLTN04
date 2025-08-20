import re
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import select, text, and_, or_
from db.database import get_db, engine
from db.models.project_tasks import project_tasks
from db.models.commits import commits
from db.models.repositories import repositories
from interfaces.task_commit_service import ITaskCommitService
import logging

logger = logging.getLogger(__name__)

class TaskCommitService(ITaskCommitService):
    """Service implementation for Task-Commit linking functionality"""
    
    def __init__(self, db: Session = None):
        self.db = db
        
    async def link_commits_to_task(
        self, 
        task_id: int, 
        repo_owner: str, 
        repo_name: str
    ) -> Dict[str, Any]:
        """Automatically link commits to a task based on commit message patterns"""
        try:
            # Get task information
            with engine.connect() as conn:
                task_query = select(project_tasks).where(project_tasks.c.id == task_id)
                task_result = conn.execute(task_query).fetchone()
                
                if not task_result:
                    return {"success": False, "message": "Task not found"}
                
                task_title = task_result.title
                assignee_username = task_result.assignee_github_username
                
                if not assignee_username:
                    return {"success": False, "message": "No assignee found for this task"}
                
                # Search for related commits
                related_commits = await self.search_commits_by_pattern(
                    repo_owner, repo_name, assignee_username, task_title
                )
                
                if not related_commits:
                    return {
                        "success": True, 
                        "message": "No matching commits found",
                        "commits_found": 0,
                        "commits": []
                    }
                
                # Update task with linked commit information
                commit_shas = [commit['sha'] for commit in related_commits]
                linked_commits_info = f"Linked commits: {', '.join(commit_shas[:5])}"  # Limit to 5 SHAs
                
                current_description = task_result.description or ""
                updated_description = f"{current_description}\n\n{linked_commits_info}".strip()
                
                # Update task description with linked commits
                from sqlalchemy import update
                update_query = update(project_tasks).where(
                    project_tasks.c.id == task_id
                ).values(
                    description=updated_description,
                    updated_at=datetime.utcnow()
                )
                conn.execute(update_query)
                conn.commit()
                
                return {
                    "success": True,
                    "message": f"Successfully linked {len(related_commits)} commits to task",
                    "commits_found": len(related_commits),
                    "commits": related_commits
                }
                
        except Exception as e:
            logger.error(f"Error linking commits to task {task_id}: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    async def get_task_related_commits(
        self, 
        task_id: int
    ) -> List[Dict[str, Any]]:
        """Get all commits related to a specific task by parsing SHAs from description"""
        try:
            with engine.connect() as conn:
                # Get task information
                task_query = select(project_tasks).where(project_tasks.c.id == task_id)
                task_result = conn.execute(task_query).fetchone()
                
                if not task_result:
                    return []
                
                repo_owner = task_result.repo_owner
                repo_name = task_result.repo_name
                description = task_result.description or ""
                
                if not all([repo_owner, repo_name]):
                    return []
                
                # Parse commit SHAs from task description
                import re
                # Look for patterns like "Linked commits:" or "Manually linked commits:" followed by SHA list
                commit_patterns = [
                    r"Linked commits:\s*([a-f0-9,\s]+)",
                    r"Manually linked commits:\s*([a-f0-9,\s]+)"
                ]
                
                commit_shas = []
                for pattern in commit_patterns:
                    matches = re.findall(pattern, description, re.IGNORECASE)
                    for match in matches:
                        # Split by comma and clean up
                        shas = [sha.strip() for sha in match.split(',') if sha.strip()]
                        commit_shas.extend(shas)
                
                # Remove duplicates and ensure valid SHA format (at least 7 chars)
                commit_shas = list(set([sha for sha in commit_shas if len(sha) >= 7]))
                
                if not commit_shas:
                    return []
                
                # Get repository ID
                repo_query = select(repositories).where(
                    and_(
                        repositories.c.owner == repo_owner,
                        repositories.c.name == repo_name
                    )
                )
                repo_result = conn.execute(repo_query).fetchone()
                
                if not repo_result:
                    return []
                
                repo_id = repo_result.id
                
                # Get commit details from database
                commits_query = select(commits).where(
                    and_(
                        commits.c.repo_id == repo_id,
                        commits.c.sha.in_(commit_shas)
                    )
                ).order_by(commits.c.committer_date.desc())
                
                results = conn.execute(commits_query).fetchall()
                
                # Convert to list of dictionaries
                commit_list = []
                for row in results:
                    commit_list.append({
                        "sha": row.sha,
                        "message": row.message,
                        "author_name": row.author_name,
                        "author_email": row.author_email,
                        "committed_date": row.committer_date.isoformat() if row.committer_date else None,
                        "insertions": row.insertions or 0,
                        "deletions": row.deletions or 0,
                        "files_changed": row.files_changed or 0,
                        "url": f"https://github.com/{repo_owner}/{repo_name}/commit/{row.sha}"
                    })
                
                return commit_list
                
        except Exception as e:
            logger.error(f"Error getting task related commits for task {task_id}: {e}")
            return []
    
    async def search_commits_by_pattern(
        self, 
        repo_owner: str, 
        repo_name: str, 
        assignee_username: str, 
        task_title: str,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Search for commits matching task pattern by a specific user"""
        try:
            with engine.connect() as conn:
                # Get repository ID
                repo_query = select(repositories).where(
                    and_(
                        repositories.c.owner == repo_owner,
                        repositories.c.name == repo_name
                    )
                )
                repo_result = conn.execute(repo_query).fetchone()
                
                if not repo_result:
                    return []
                
                repo_id = repo_result.id
                
                # Calculate date range
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=days_back)
                
                # Create search patterns
                patterns = self._create_search_patterns(task_title)
                
                # Search commits with multiple patterns
                commit_conditions = []
                for pattern in patterns:
                    commit_conditions.append(commits.c.message.ilike(f"%{pattern}%"))
                
                # Build query
                commits_query = select(commits).where(
                    and_(
                        commits.c.repo_id == repo_id,
                        commits.c.author_name == assignee_username,
                        commits.c.committer_date >= start_date,
                        commits.c.committer_date <= end_date,
                        or_(*commit_conditions) if commit_conditions else True
                    )
                ).order_by(commits.c.committer_date.desc())
                
                results = conn.execute(commits_query).fetchall()
                
                # Convert to list of dictionaries
                commit_list = []
                for row in results:
                    commit_list.append({
                        "sha": row.sha,
                        "message": row.message,
                        "author_name": row.author_name,
                        "author_email": row.author_email,
                        "committed_date": row.committer_date.isoformat() if row.committer_date else None,
                        "insertions": row.insertions,
                        "deletions": row.deletions,
                        "files_changed": row.files_changed,
                        "url": f"https://github.com/{repo_owner}/{repo_name}/commit/{row.sha}"
                    })
                
                return commit_list
                
        except Exception as e:
            logger.error(f"Error searching commits by pattern: {e}")
            return []
    
    async def get_user_recent_commits(
        self, 
        repo_owner: str, 
        repo_name: str, 
        username: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recent commits by a specific user in a repository"""
        try:
            with engine.connect() as conn:
                # Get repository ID
                repo_query = select(repositories).where(
                    and_(
                        repositories.c.owner == repo_owner,
                        repositories.c.name == repo_name
                    )
                )
                repo_result = conn.execute(repo_query).fetchone()
                
                if not repo_result:
                    return []
                
                repo_id = repo_result.id
                
                # Get recent commits by user
                commits_query = select(commits).where(
                    and_(
                        commits.c.repo_id == repo_id,
                        commits.c.author_name == username
                    )
                ).order_by(commits.c.committer_date.desc()).limit(limit)
                
                results = conn.execute(commits_query).fetchall()
                
                # Convert to list of dictionaries
                commit_list = []
                for row in results:
                    commit_list.append({
                        "sha": row.sha,
                        "message": row.message,
                        "author_name": row.author_name,
                        "author_email": row.author_email,
                        "committed_date": row.committer_date.isoformat() if row.committer_date else None,
                        "insertions": row.insertions,
                        "deletions": row.deletions,
                        "files_changed": row.files_changed,
                        "url": f"https://github.com/{repo_owner}/{repo_name}/commit/{row.sha}"
                    })
                
                return commit_list
                
        except Exception as e:
            logger.error(f"Error getting user recent commits: {e}")
            return []
    
    def _create_search_patterns(self, task_title: str) -> List[str]:
        """Create search patterns based on task title"""
        patterns = []
        
        # Clean task title
        cleaned_title = re.sub(r'[^\w\s]', '', task_title.lower())
        words = cleaned_title.split()
        
        # Pattern 1: "Task [title]"
        patterns.append(f"task {task_title}")
        patterns.append(f"Task {task_title}")
        
        # Pattern 2: "[title] task"
        patterns.append(f"{task_title} task")
        
        # Pattern 3: Individual words from title
        if len(words) > 1:
            for word in words:
                if len(word) > 3:  # Only meaningful words
                    patterns.append(word)
        
        # Pattern 4: First few words of title
        if len(words) >= 2:
            patterns.append(" ".join(words[:2]))
        
        # Pattern 5: Variations with common prefixes
        prefixes = ["fix", "feat", "feature", "implement", "add", "update"]
        for prefix in prefixes:
            patterns.append(f"{prefix} {task_title}")
        
        return patterns[:10]  # Limit to 10 patterns to avoid too broad search

    async def get_user_commits_with_pagination(
        self, 
        repo_owner: str, 
        repo_name: str, 
        username: str,
        limit: int = 10,
        offset: int = 0,
        search: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get user commits with pagination and search support"""
        try:
            with engine.connect() as conn:
                # Get repository ID
                repo_query = select(repositories).where(
                    and_(
                        repositories.c.owner == repo_owner,
                        repositories.c.name == repo_name
                    )
                )
                repo_result = conn.execute(repo_query).fetchone()
                
                if not repo_result:
                    return {"commits": [], "total": 0, "has_more": False}
                
                repo_id = repo_result.id
                
                # Build base query
                base_conditions = [
                    commits.c.repo_id == repo_id,
                    commits.c.author_name == username
                ]
                
                # Add search condition if provided
                if search and search.strip():
                    search_term = f"%{search.strip()}%"
                    search_conditions = or_(
                        commits.c.message.ilike(search_term),
                        commits.c.sha.ilike(search_term),
                        commits.c.author_name.ilike(search_term),
                        commits.c.author_email.ilike(search_term)
                    )
                    base_conditions.append(search_conditions)
                
                # Count total commits
                count_query = select(commits.c.id).where(and_(*base_conditions))
                total_result = conn.execute(count_query).fetchall()
                total_commits = len(total_result)
                
                # Get paginated commits
                commits_query = select(commits).where(
                    and_(*base_conditions)
                ).order_by(commits.c.committer_date.desc()).limit(limit).offset(offset)
                
                results = conn.execute(commits_query).fetchall()
                
                # Convert to list of dictionaries
                commit_list = []
                for row in results:
                    commit_list.append({
                        "sha": row.sha,
                        "message": row.message,
                        "author_name": row.author_name,
                        "author_email": row.author_email,
                        "committed_date": row.committer_date.isoformat() if row.committer_date else None,
                        "insertions": row.insertions or 0,
                        "deletions": row.deletions or 0,
                        "files_changed": row.files_changed or 0,
                        "url": f"https://github.com/{repo_owner}/{repo_name}/commit/{row.sha}"
                    })
                
                has_more = (offset + limit) < total_commits
                
                return {
                    "commits": commit_list,
                    "total": total_commits,
                    "has_more": has_more
                }
                
        except Exception as e:
            logger.error(f"Error getting user commits with pagination: {e}")
            return {"commits": [], "total": 0, "has_more": False}

    async def link_specific_commits(
        self, 
        task_id: int, 
        repo_owner: str, 
        repo_name: str, 
        commit_shas: List[str]
    ) -> Dict[str, Any]:
        """Link specific commits to a task by their SHA values"""
        try:
            with engine.connect() as conn:
                # Get task information
                task_query = select(project_tasks).where(project_tasks.c.id == task_id)
                task_result = conn.execute(task_query).fetchone()
                
                if not task_result:
                    return {"success": False, "message": "Task not found"}
                
                # Get repository ID
                repo_query = select(repositories).where(
                    and_(
                        repositories.c.owner == repo_owner,
                        repositories.c.name == repo_name
                    )
                )
                repo_result = conn.execute(repo_query).fetchone()
                
                if not repo_result:
                    return {"success": False, "message": "Repository not found"}
                
                repo_id = repo_result.id
                
                # Verify commits exist
                commits_query = select(commits).where(
                    and_(
                        commits.c.repo_id == repo_id,
                        commits.c.sha.in_(commit_shas)
                    )
                )
                found_commits = conn.execute(commits_query).fetchall()
                
                if not found_commits:
                    return {"success": False, "message": "No valid commits found"}
                
                # Update task with linked commit information
                found_shas = [commit.sha for commit in found_commits]
                linked_commits_info = f"Manually linked commits: {', '.join(found_shas)}"
                
                current_description = task_result.description or ""
                
                # Check if there are already linked commits in description
                if "Linked commits:" in current_description or "Manually linked commits:" in current_description:
                    # Replace existing linked commits info
                    lines = current_description.split('\n')
                    filtered_lines = [line for line in lines if not (
                        "Linked commits:" in line or "Manually linked commits:" in line
                    )]
                    updated_description = '\n'.join(filtered_lines + [linked_commits_info]).strip()
                else:
                    # Append new linked commits info
                    updated_description = f"{current_description}\n\n{linked_commits_info}".strip()
                
                # Update task description with linked commits
                from sqlalchemy import update
                update_query = update(project_tasks).where(
                    project_tasks.c.id == task_id
                ).values(
                    description=updated_description,
                    updated_at=datetime.utcnow()
                )
                conn.execute(update_query)
                conn.commit()
                
                # Convert found commits to response format
                commit_list = []
                for commit in found_commits:
                    commit_list.append({
                        "sha": commit.sha,
                        "message": commit.message,
                        "author_name": commit.author_name,
                        "author_email": commit.author_email,
                        "committed_date": commit.committer_date.isoformat() if commit.committer_date else None,
                        "insertions": commit.insertions or 0,
                        "deletions": commit.deletions or 0,
                        "files_changed": commit.files_changed or 0,
                        "url": f"https://github.com/{repo_owner}/{repo_name}/commit/{commit.sha}"
                    })
                
                return {
                    "success": True,
                    "message": f"Successfully linked {len(found_commits)} commits to task",
                    "commits_found": len(found_commits),
                    "commits": commit_list
                }
                
        except Exception as e:
            logger.error(f"Error linking specific commits to task {task_id}: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    async def get_repository_authors(
        self, 
        repo_owner: str, 
        repo_name: str
    ) -> List[Dict[str, Any]]:
        """Get all authors in a repository with commit statistics"""
        try:
            with engine.connect() as conn:
                # Get repository ID
                repo_query = select(repositories).where(
                    and_(
                        repositories.c.owner == repo_owner,
                        repositories.c.name == repo_name
                    )
                )
                repo_result = conn.execute(repo_query).fetchone()
                
                if not repo_result:
                    return []
                
                repo_id = repo_result.id
                
                # Get authors with statistics
                authors_query = text("""
                    SELECT 
                        author_name,
                        author_email,
                        COUNT(*) as commit_count,
                        MAX(committer_date) as last_commit_date,
                        SUM(COALESCE(insertions, 0)) as total_insertions,
                        SUM(COALESCE(deletions, 0)) as total_deletions,
                        SUM(COALESCE(files_changed, 0)) as total_files_changed
                    FROM commits 
                    WHERE repo_id = :repo_id 
                    GROUP BY author_name, author_email
                    ORDER BY commit_count DESC, last_commit_date DESC
                """)
                
                results = conn.execute(authors_query, {"repo_id": repo_id}).fetchall()
                
                # Convert to list of dictionaries
                authors_list = []
                for row in results:
                    authors_list.append({
                        "author_name": row.author_name,
                        "author_email": row.author_email,
                        "commit_count": row.commit_count,
                        "last_commit_date": row.last_commit_date.isoformat() if row.last_commit_date else None,
                        "total_insertions": row.total_insertions or 0,
                        "total_deletions": row.total_deletions or 0,
                        "total_files_changed": row.total_files_changed or 0,
                        "display_name": f"{row.author_name} ({row.commit_count} commits)"
                    })
                
                return authors_list
                
        except Exception as e:
            logger.error(f"Error getting repository authors: {e}")
            return []
