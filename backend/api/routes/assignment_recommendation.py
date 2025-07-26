# backend/api/routes/assignment_recommendation.py
"""
Assignment Recommendation API Routes
Cung cấp các endpoints để đề xuất phân công thành viên dựa trên AI analysis
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from pydantic import BaseModel
from db.database import get_db
from db.models.repositories import repositories
from services.assignment_recommendation_service import AssignmentRecommendationService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assignment-recommendation", tags=["Assignment Recommendation"])

# Pydantic models
class TaskRequirement(BaseModel):
    task_type: str  # feat, fix, docs, refactor, etc.
    task_area: str  # frontend, backend, database, etc.
    risk_level: str  # low, medium, high
    priority: Optional[str] = "MEDIUM"  # LOW, MEDIUM, HIGH, URGENT
    required_skills: Optional[List[str]] = None
    exclude_members: Optional[List[str]] = None
    description: Optional[str] = None

class RecommendationResponse(BaseModel):
    member: str
    score: float
    adjusted_score: Optional[float] = None
    explanation: str
    profile_summary: Dict[str, Any]
    workload_info: Optional[Dict[str, Any]] = None

class SkillAnalysisResponse(BaseModel):
    member: str
    total_commits: int
    expertise_areas: List[str]
    risk_tolerance: str
    recent_activity_score: float
    top_commit_types: Dict[str, int]
    top_areas: Dict[str, int]
    top_languages: Dict[str, int]

# ==================== SIMPLIFIED ENDPOINTS FOR FRONTEND ====================

class SmartAssignRequest(BaseModel):
    task_description: str
    required_skills: Optional[List[str]] = []
    consider_workload: Optional[bool] = True

@router.get("/test/{owner}/{repo_name}")
async def test_endpoint(
    owner: str,
    repo_name: str,
    db: Session = Depends(get_db)
):
    """
    Test endpoint for debugging
    """
    try:
        # Get repository from owner/repo_name
        query = select(repositories).where(
            repositories.c.owner == owner,
            repositories.c.name == repo_name
        )
        result = db.execute(query).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo_name} not found")
        
        repo_id = result.id
        
        service = AssignmentRecommendationService(db)
        member_skills = service.analyze_member_skills(repo_id, 90)
        
        return {
            "repo_id": repo_id,
            "member_skills_type": str(type(member_skills)),
            "member_count": len(member_skills) if hasattr(member_skills, '__len__') else 0,
            "first_member": list(member_skills.keys())[0] if member_skills and hasattr(member_skills, 'keys') else None
        }
        
    except Exception as e:
        logger.error(f"Error in test endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/member-skills/{owner}/{repo_name}")
async def get_member_skills_simple(
    owner: str,
    repo_name: str,
    db: Session = Depends(get_db)
):
    """
    Simplified member skills endpoint for frontend
    """
    try:
        # Get repository from owner/repo_name
        query = select(repositories).where(
            repositories.c.owner == owner,
            repositories.c.name == repo_name
        )
        result = db.execute(query).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo_name} not found")
        
        repo_id = result.id
        
        service = AssignmentRecommendationService(db)
        member_skills = service.analyze_member_skills(repo_id, 90)
        
        # Ensure member_skills is a dictionary
        if not isinstance(member_skills, dict):
            logger.error(f"member_skills is not a dict: {type(member_skills)}")
            member_skills = {}
        
        # Format for frontend
        members = []
        for username, profile in member_skills.items():
            if not isinstance(profile, dict) or profile.get('total_commits', 0) == 0:
                continue
                
            # Get top skills with confidence scores
            skills = []
            expertise_areas = profile.get('expertise_areas', [])
            
            # Handle expertise_areas as list (from service)
            if isinstance(expertise_areas, list):
                for skill in expertise_areas:
                    # Get confidence from area counts
                    area_counts = profile.get('areas', {})
                    confidence = area_counts.get(skill, 0) / max(profile.get('total_commits', 1), 1)
                    skills.append({
                        'skill': skill,
                        'confidence': float(confidence)
                    })
            elif isinstance(expertise_areas, dict):
                for skill, confidence in expertise_areas.items():
                    skills.append({
                        'skill': skill,
                        'confidence': float(confidence)
                    })
            
            # Add top commit types as skills
            commit_types = profile.get('commit_types', {})
            for commit_type, count in sorted(commit_types.items(), key=lambda x: x[1], reverse=True)[:3]:
                if count >= 2:  # Only include if significant
                    confidence = count / max(profile.get('total_commits', 1), 1)
                    skills.append({
                        'skill': f"commit_{commit_type}",
                        'confidence': float(confidence)
                    })
            
            # Add top languages as skills
            languages = profile.get('languages', {})
            for language, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:3]:
                if count >= 2:  # Only include if significant
                    confidence = count / max(profile.get('total_commits', 1), 1)
                    skills.append({
                        'skill': f"lang_{language}",
                        'confidence': float(confidence)
                    })
            
            # Sort skills by confidence
            skills.sort(key=lambda x: x['confidence'], reverse=True)
            
            members.append({
                'username': username,
                'display_name': username,  # Could enhance this later
                'avatar_url': f"https://github.com/{username}.png",
                'total_commits': profile.get('total_commits', 0),
                'skills': skills[:10],  # Top 10 skills
                'recent_activity_score': round(profile.get('recent_activity_score', 0), 2)
            })
        
        # Sort members by total commits
        members.sort(key=lambda x: x['total_commits'], reverse=True)
        
        return {
            'members': members,
            'total_members': len(members)
        }
        
    except Exception as e:
        logger.error(f"Error getting member skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/member-skills-simple/{owner}/{repo_name}")
async def get_member_skills_simple_alias(
    owner: str,
    repo_name: str,
    db: Session = Depends(get_db)
):
    """
    Alias endpoint for frontend compatibility
    """
    return await get_member_skills_simple(owner, repo_name, db)

@router.post("/smart-assign/{owner}/{repo_name}")
async def smart_assign_simple(
    owner: str,
    repo_name: str,
    request: SmartAssignRequest,
    db: Session = Depends(get_db)
):
    """
    Simplified smart assignment endpoint for frontend
    """
    try:
        # Get repository
        query = select(repositories).where(
            repositories.c.owner == owner,
            repositories.c.name == repo_name
        )
        result = db.execute(query).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo_name} not found")
        
        repo_id = result.id
        
        service = AssignmentRecommendationService(db)
        
        # Analyze task description to determine task characteristics
        task_type = "feature"  # Default
        task_area = "general"  # Default  
        risk_level = "medium"  # Default
        
        # Simple heuristics to determine task characteristics from description
        description_lower = request.task_description.lower()
        if any(word in description_lower for word in ["fix", "bug", "error", "issue"]):
            task_type = "fix"
        elif any(word in description_lower for word in ["test", "testing", "spec"]):
            task_type = "test"
        elif any(word in description_lower for word in ["doc", "documentation", "readme"]):
            task_type = "docs"
        elif any(word in description_lower for word in ["refactor", "optimize", "clean"]):
            task_type = "refactor"
        
        if any(word in description_lower for word in ["frontend", "ui", "interface", "react", "component"]):
            task_area = "frontend"
        elif any(word in description_lower for word in ["backend", "api", "server", "database"]):
            task_area = "backend"
        elif any(word in description_lower for word in ["database", "sql", "query"]):
            task_area = "database"
        
        # Get recommendations
        recommendations = service.recommend_with_workload_balance(
            repository_id=repo_id,
            task_type=task_type,
            task_area=task_area,
            risk_level=risk_level,
            task_priority="MEDIUM",
            required_skills=request.required_skills or [],
            top_k=5
        )
        
        return {
            "recommendations": recommendations,
            "task_analysis": {
                "detected_type": task_type,
                "detected_area": task_area,
                "risk_level": risk_level
            }
        }
        
    except Exception as e:
        logger.error(f"Error in smart assignment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/team-insights/{owner}/{repo_name}")
async def get_team_insights(
    owner: str,
    repo_name: str,
    days_back: int = Query(90, description="Số ngày quay lại để phân tích"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Phân tích insights tổng thể về team
    """
    try:
        # Get repository from owner/repo_name
        query = select(repositories).where(
            repositories.c.owner == owner,
            repositories.c.name == repo_name
        )
        result = db.execute(query).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo_name} not found")
        
        repo_id = result.id
        
        service = AssignmentRecommendationService(db)
        member_skills = service.analyze_member_skills(repo_id, days_back)
        
        # Ensure member_skills is a dictionary
        if not isinstance(member_skills, dict):
            logger.error(f"member_skills is not a dict: {type(member_skills)}")
            member_skills = {}
        
        # Calculate team insights
        active_profiles = {m: p for m, p in member_skills.items() if isinstance(p, dict) and p.get('total_commits', 0) > 0}
        total_members = len(active_profiles)
        total_commits = sum(p.get('total_commits', 0) for p in active_profiles.values())
        
        # Top skills across team
        all_skills = {}
        for profile in active_profiles.values():
            if not isinstance(profile, dict):
                continue
            expertise_areas = profile.get('expertise_areas', {})
            if isinstance(expertise_areas, dict):
                for skill, confidence in expertise_areas.items():
                    if skill not in all_skills:
                        all_skills[skill] = {'total_confidence': 0, 'member_count': 0}
                    all_skills[skill]['total_confidence'] += confidence
                    all_skills[skill]['member_count'] += 1
        
        top_skills = sorted(
            [{'skill': skill, 'member_count': data['member_count'], 'avg_confidence': data['total_confidence']/data['member_count']} 
             for skill, data in all_skills.items()],
            key=lambda x: (x['member_count'], x['avg_confidence']),
            reverse=True
        )[:10]
        
        # Top contributors
        top_contributors = sorted(
            [{'username': member, 'commits': profile.get('total_commits', 0)} 
             for member, profile in active_profiles.items()],
            key=lambda x: x['commits'],
            reverse=True
        )[:5]
        
        return {
            "total_members": total_members,
            "active_members": len([m for m, p in active_profiles.items() if p.get('recent_activity_score', 0) > 0.1]),
            "total_commits": total_commits,
            "avg_commits_per_member": round(total_commits / max(total_members, 1), 1),
            "total_skills": len(all_skills),
            "top_skills": top_skills,
            "top_contributors": top_contributors
        }
        
    except Exception as e:
        logger.error(f"Error getting team insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workload-analysis/{owner}/{repo_name}")
async def get_workload_analysis(
    owner: str,
    repo_name: str,
    days_back: int = Query(30, description="Số ngày quay lại để phân tích"),
    db: Session = Depends(get_db)
):
    """
    Phân tích workload của các thành viên
    """
    try:
        # Get repository from owner/repo_name
        query = select(repositories).where(
            repositories.c.owner == owner,
            repositories.c.name == repo_name
        )
        result = db.execute(query).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo_name} not found")
        
        repo_id = result.id
        
        service = AssignmentRecommendationService(db)
        workload = service.get_member_workload(repo_id, days_back)
        
        return workload
        
    except Exception as e:
        logger.error(f"Error getting workload analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ALIAS ENDPOINTS FOR FRONTEND COMPATIBILITY ====================

@router.post("/smart-assign-simple/{owner}/{repo_name}")
async def smart_assign_simple_alias(
    owner: str,
    repo_name: str,
    request: SmartAssignRequest,
    db: Session = Depends(get_db)
):
    """
    Alias endpoint for frontend compatibility
    """
    return await smart_assign_simple(owner, repo_name, request, db)

@router.get("/team-insights-simple/{owner}/{repo_name}")
async def get_team_insights_simple_alias(
    owner: str,
    repo_name: str,
    days_back: int = Query(90, description="Số ngày quay lại để phân tích"),
    db: Session = Depends(get_db)
):
    """
    Alias endpoint for frontend compatibility
    """
    return await get_team_insights(owner, repo_name, days_back, db)
