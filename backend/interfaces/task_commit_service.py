from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

class ITaskCommitService(ABC):
    """Interface for Task-Commit linking service"""
    
    @abstractmethod
    async def link_commits_to_task(
        self, 
        task_id: int, 
        repo_owner: str, 
        repo_name: str
    ) -> Dict[str, Any]:
        """Automatically link commits to a task based on commit message patterns"""
        pass
    
    @abstractmethod
    async def get_task_related_commits(
        self, 
        task_id: int
    ) -> List[Dict[str, Any]]:
        """Get all commits related to a specific task"""
        pass
    
    @abstractmethod
    async def search_commits_by_pattern(
        self, 
        repo_owner: str, 
        repo_name: str, 
        assignee_username: str, 
        task_title: str,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Search for commits matching task pattern by a specific user"""
        pass
    
    @abstractmethod
    async def get_user_recent_commits(
        self, 
        repo_owner: str, 
        repo_name: str, 
        username: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recent commits by a specific user in a repository"""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def link_specific_commits(
        self, 
        task_id: int, 
        repo_owner: str, 
        repo_name: str, 
        commit_shas: List[str]
    ) -> Dict[str, Any]:
        """Link specific commits to a task by their SHA values"""
        pass

    @abstractmethod
    async def get_repository_authors(
        self, 
        repo_owner: str, 
        repo_name: str
    ) -> List[Dict[str, Any]]:
        """Get all authors in a repository with commit statistics"""
        pass
