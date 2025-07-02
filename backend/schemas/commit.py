from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any


class CommitCreate(BaseModel):
    commit_id: str = Field(..., description="Unique commit SHA")
    message: str = Field(..., description="Commit message")
    author_name: str = Field(..., description="Author name")
    author_email: str = Field(..., description="Author email")
    committed_date: datetime = Field(..., description="Commit date")
    repository_id: int = Field(..., description="Repository ID")
    
    # Code change statistics
    insertions: Optional[int] = Field(None, description="Number of lines added")
    deletions: Optional[int] = Field(None, description="Number of lines deleted")
    files_changed: Optional[int] = Field(None, description="Number of files changed")
    
    # Enhanced file tracking
    modified_files: Optional[List[str]] = Field(None, description="List of modified file paths")
    file_types: Optional[Dict[str, int]] = Field(None, description="File extensions and their counts")
    modified_directories: Optional[Dict[str, int]] = Field(None, description="Directories and their change counts")
    total_changes: Optional[int] = Field(None, description="Total changes (additions + deletions)")
    
    # Metadata
    is_merge: Optional[bool] = Field(False, description="Whether this is a merge commit")
    change_type: Optional[str] = Field(None, description="Type of change (feature, bugfix, refactor, etc.)")
    commit_size: Optional[str] = Field(None, description="Size category (small, medium, large)")
    
    # Additional commit details
    parent_sha: Optional[str] = Field(None, description="Parent commit SHA")
    branch_name: Optional[str] = Field(None, description="Branch name")
    merge_from_branch: Optional[str] = Field(None, description="Source branch for merge commits")


class CommitOut(CommitCreate):
    id: int
    created_at: Optional[datetime] = None
    last_synced: Optional[datetime] = None

    class Config:
        from_attributes = True  # For Pydantic V2


class CommitStats(BaseModel):
    """Schema for commit statistics response"""
    total_commits: int
    total_additions: int
    total_deletions: int
    total_files_changed: int
    file_type_distribution: Dict[str, int]
    directory_distribution: Dict[str, int]
    commit_size_distribution: Dict[str, int]
    change_type_distribution: Dict[str, int]


class CommitAnalysis(BaseModel):
    """Schema for commit analysis response"""
    commit: CommitOut
    analysis: Dict[str, Any]
    recommendations: List[str]
