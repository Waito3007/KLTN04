# backend/api/routes/ai.py
"""
AI Routes - API endpoints for HAN-based AI features
Provides endpoints for commit analysis, task assignment, and project insights
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from services.han_ai_service import get_han_ai_service
from services.commit_service import get_commits_by_repo
from services.repo_service import get_repo_by_owner_and_name

logger = logging.getLogger(__name__)

ai_router = APIRouter(prefix="/ai", tags=["AI & Machine Learning"])

# Pydantic models for request/response
class CommitAnalysisRequest(BaseModel):
    message: str = Field(..., description="Commit message to analyze")

class BatchCommitAnalysisRequest(BaseModel):
    messages: List[str] = Field(..., description="List of commit messages to analyze")

class CommitAnalysisResponse(BaseModel):
    success: bool
    message: str
    analysis: Dict[str, Any]
    model_version: Optional[str] = None
    confidence: Optional[Dict[str, float]] = None

class TaskAssignmentRequest(BaseModel):
    tasks: List[Dict[str, Any]] = Field(..., description="List of tasks to assign")
    developers: List[Dict[str, Any]] = Field(..., description="List of available developers")

class ProjectInsightsRequest(BaseModel):
    project_data: Dict[str, Any] = Field(..., description="Project data for analysis")

class DeveloperPattern(BaseModel):
    developer: str
    commits: List[str]

class DeveloperPatternsRequest(BaseModel):
    developer_commits: List[DeveloperPattern] = Field(..., description="Developer commit patterns")

# AI Analysis Endpoints
@ai_router.post("/analyze-commit", response_model=CommitAnalysisResponse)
async def analyze_commit(request: CommitAnalysisRequest):
    """
    Analyze a single commit message using HAN model
    
    Provides:
    - Commit category classification (bug, feature, docs, etc.)
    - Impact assessment (low, medium, high)
    - Urgency evaluation (low, medium, high)
    - Confidence scores for predictions
    """
    try:
        ai_service = get_han_ai_service()
        result = await ai_service.analyze_commit_message(request.message)
        
        return CommitAnalysisResponse(**result)
        
    except Exception as e:
        logger.error(f"Error analyzing commit: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@ai_router.post("/analyze-commits-batch")
async def analyze_commits_batch(request: BatchCommitAnalysisRequest):
    """
    Analyze multiple commit messages in batch
    
    Provides:
    - Batch analysis of all commits
    - Statistical summary of commit patterns
    - Distribution of categories, impacts, and urgencies
    """
    try:
        ai_service = get_han_ai_service()
        result = await ai_service.analyze_commits_batch(request.messages)
        
        return {
            "success": True,
            "data": result,
            "message": f"Analyzed {len(request.messages)} commits successfully"
        }
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@ai_router.get("/analyze-repo/{owner}/{repo}")
async def analyze_repository_commits(
    owner: str, 
    repo: str,
    limit: int = 100
):
    """
    Analyze all commits in a repository
    
    Provides:
    - Complete commit analysis for the repository
    - Repository-specific insights and patterns
    - Quality metrics and trends
    """
    try:
        # Get commits from database
        commits = await get_commits_by_repo(owner, repo, limit)
        
        if not commits:
            raise HTTPException(status_code=404, detail="No commits found for this repository")
        
        # Extract commit messages
        commit_messages = [commit.message for commit in commits if commit.message]
        
        ai_service = get_han_ai_service()
        result = await ai_service.analyze_commits_batch(commit_messages)
        
        # Add repository context
        result['repository'] = f"{owner}/{repo}"
        result['analyzed_commits'] = len(commit_messages)
        
        return {
            "success": True,
            "data": result,
            "message": f"Repository {owner}/{repo} analysis completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing repository: {e}")
        raise HTTPException(status_code=500, detail=f"Repository analysis failed: {str(e)}")

# Developer Analysis Endpoints
@ai_router.post("/analyze-developer-patterns")
async def analyze_developer_patterns(request: DeveloperPatternsRequest):
    """
    Analyze commit patterns for developers
    
    Provides:
    - Individual developer profiles based on commit history
    - Specialization identification (bug-fixer, feature-developer, etc.)
    - Activity and expertise scores
    """
    try:
        # Convert request to expected format
        developer_commits = {
            pattern.developer: pattern.commits 
            for pattern in request.developer_commits
        }
        
        ai_service = get_han_ai_service()
        result = await ai_service.analyze_developer_patterns(developer_commits)
        
        return {
            "success": True,
            "data": result,
            "message": f"Analyzed patterns for {len(developer_commits)} developers"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing developer patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Developer analysis failed: {str(e)}")

@ai_router.get("/developer-profile/{owner}/{repo}/{developer}")
async def get_developer_profile(owner: str, repo: str, developer: str):
    """
    Get AI-generated profile for a specific developer in a repository
    
    Provides:
    - Developer specialization analysis
    - Commit pattern insights
    - Productivity and focus area recommendations
    """
    try:
        # Get commits by this developer
        commits = await get_commits_by_repo(owner, repo, limit=200)
        developer_commits = [
            commit.message for commit in commits 
            if commit.author == developer and commit.message
        ]
        
        if not developer_commits:
            raise HTTPException(status_code=404, detail=f"No commits found for developer {developer}")
        
        ai_service = get_han_ai_service()
        result = await ai_service.analyze_developer_patterns({developer: developer_commits})
        
        return {
            "success": True,
            "developer": developer,
            "repository": f"{owner}/{repo}",
            "profile": result['developer_profiles'].get(developer, {}),
            "commit_count": len(developer_commits)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting developer profile: {e}")
        raise HTTPException(status_code=500, detail=f"Profile generation failed: {str(e)}")

# Task Management Endpoints
@ai_router.post("/suggest-task-assignment")
async def suggest_task_assignment(request: TaskAssignmentRequest):
    """
    Suggest optimal task assignments based on AI analysis
    
    Provides:
    - AI-powered task-to-developer matching
    - Confidence scores for assignments
    - Reasoning behind each recommendation
    """
    try:
        ai_service = get_han_ai_service()
        result = await ai_service.suggest_task_assignment(request.tasks, request.developers)
        
        return {
            "success": True,
            "data": result,
            "message": f"Generated assignments for {len(request.tasks)} tasks"
        }
        
    except Exception as e:
        logger.error(f"Error in task assignment: {e}")
        raise HTTPException(status_code=500, detail=f"Task assignment failed: {str(e)}")

@ai_router.get("/recommend-tasks/{owner}/{repo}")
async def recommend_tasks_for_repo(owner: str, repo: str):
    """
    Recommend task priorities and assignments for a repository
    
    Provides:
    - Repository-specific task recommendations
    - Priority suggestions based on commit analysis
    - Developer assignment recommendations
    """
    try:
        # Get repository data
        repo_data = await get_repo_by_owner_and_name(owner, repo)
        if not repo_data:
            raise HTTPException(status_code=404, detail="Repository not found")
        
        # Get recent commits for analysis
        commits = await get_commits_by_repo(owner, repo, limit=100)
        
        # Create project data structure
        project_data = {
            'name': f"{owner}/{repo}",
            'commits': [{'message': c.message, 'author': c.author} for c in commits],
            'contributors': list(set(c.author for c in commits if c.author))
        }
        
        ai_service = get_han_ai_service()
        insights = await ai_service.generate_project_insights(project_data)
        
        return {
            "success": True,
            "repository": f"{owner}/{repo}",
            "recommendations": insights,
            "message": "Task recommendations generated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating task recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

# Project Insights Endpoints
@ai_router.post("/project-insights")
async def generate_project_insights(request: ProjectInsightsRequest):
    """
    Generate comprehensive AI-powered project insights
    
    Provides:
    - Overall project health assessment
    - Code quality trends analysis
    - Team collaboration patterns
    - Actionable recommendations
    """
    try:
        ai_service = get_han_ai_service()
        result = await ai_service.generate_project_insights(request.project_data)
        
        return {
            "success": True,
            "data": result,
            "message": "Project insights generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error generating project insights: {e}")
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")

@ai_router.get("/insights/{owner}/{repo}")
async def get_repository_insights(owner: str, repo: str):
    """
    Get comprehensive insights for a specific repository
    
    Provides:
    - Repository health metrics
    - Development pattern analysis
    - Quality assessment and recommendations
    - Team productivity insights
    """
    try:
        # Get repository and commit data
        commits = await get_commits_by_repo(owner, repo, limit=200)
        
        if not commits:
            raise HTTPException(status_code=404, detail="No data found for analysis")
        
        # Prepare project data
        commit_data = []
        contributors = set()
        
        for commit in commits:
            if commit.message:
                commit_data.append({
                    'message': commit.message,
                    'author': commit.author,
                    'sha': commit.sha,
                    'date': commit.date.isoformat() if commit.date else None
                })
                if commit.author:
                    contributors.add(commit.author)
        
        project_data = {
            'name': f"{owner}/{repo}",
            'commits': commit_data,
            'contributors': list(contributors)
        }
        
        ai_service = get_han_ai_service()
        insights = await ai_service.generate_project_insights(project_data)
        
        return {
            "success": True,
            "repository": f"{owner}/{repo}",
            "total_commits_analyzed": len(commit_data),
            "total_contributors": len(contributors),
            "insights": insights,
            "message": "Repository insights generated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting repository insights: {e}")
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")

# Model Management Endpoints
@ai_router.get("/model/status")
async def get_model_status():
    """
    Get current status of the HAN AI model
    
    Provides:
    - Model loading status
    - Version information
    - Performance metrics
    """
    try:
        ai_service = get_han_ai_service()
        
        return {
            "success": True,
            "model_loaded": ai_service.is_model_loaded,
            "model_type": "HAN (Hierarchical Attention Network)",
            "version": "1.0",
            "capabilities": [
                "Commit message classification",
                "Impact assessment", 
                "Urgency evaluation",
                "Developer pattern analysis",
                "Task assignment suggestions",
                "Project insights generation"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@ai_router.post("/model/warm-up")
async def warm_up_model():
    """
    Warm up the HAN model for faster subsequent predictions
    """
    try:
        ai_service = get_han_ai_service()
        
        # Run a test prediction to warm up the model
        test_result = await ai_service.analyze_commit_message("feat: add new feature for testing")
        
        return {
            "success": True,
            "message": "Model warmed up successfully",
            "test_prediction": test_result.get('analysis', {})
        }
        
    except Exception as e:
        logger.error(f"Error warming up model: {e}")
        raise HTTPException(status_code=500, detail=f"Model warm-up failed: {str(e)}")

# Health check endpoint
@ai_router.get("/health")
async def ai_health_check():
    """
    Health check for AI services
    """
    try:
        ai_service = get_han_ai_service()
        
        return {
            "status": "healthy",
            "model_available": ai_service.is_model_loaded,
            "service": "HAN AI Service",
            "endpoints_available": 12
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "HAN AI Service"
        }
