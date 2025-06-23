from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from db.database import get_db
from services.member_analysis_service import MemberAnalysisService

router = APIRouter(prefix="/api/repositories", tags=["member-analysis"])

@router.get("/{repo_id}/members")
async def get_repository_members(
    repo_id: int,
    db: Session = Depends(get_db)
):
    """Lấy danh sách members của repository"""
    try:
        service = MemberAnalysisService(db)
        members = service.get_repository_members(repo_id)
        
        return {
            "success": True,
            "repository_id": repo_id,
            "members": members,
            "total": len(members)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching members: {str(e)}")

@router.get("/{repo_id}/members/{member_login}/commits")
async def get_member_commits_analysis(
    repo_id: int,
    member_login: str,
    branch_name: str = None,  # NEW: Optional branch filter
    limit: int = 50,
    use_ai: bool = True,
    db: Session = Depends(get_db)
):
    """Lấy commits của member với AI analysis và branch filter"""
    try:
        service = MemberAnalysisService(db)
        
        if use_ai:
            # Use AI-powered analysis
            analysis = await service.get_member_commits_with_ai_analysis(
                repo_id, member_login, limit, branch_name
            )
        else:
            # Use pattern-based analysis
            analysis = service.get_member_commits_with_analysis(
                repo_id, member_login, limit, branch_name
            )
        
        return {
            "success": True,
            "data": analysis,
            "branch_filter": branch_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing member commits: {str(e)}")

@router.get("/{repo_id}/ai-features")
async def get_ai_features_status(repo_id: int):
    """Lấy status của các tính năng AI available"""
    return {
        "success": True,
        "repository_id": repo_id,
        "features": {
            "commit_analysis": True,
            "member_insights": True,
            "productivity_tracking": True,
            "code_pattern_detection": True,
            "han_model_analysis": True
        },
        "ai_model": {
            "name": "HAN Commit Analyzer",
            "version": "1.0",
            "type": "Hierarchical Attention Network",
            "capabilities": [
                "Deep commit message understanding",
                "Semantic commit classification",
                "Developer behavior analysis", 
                "Technology area detection",
                "Impact and urgency assessment",
                "Code quality insights"
            ]
        },
        "endpoints": {
            "commit_analysis": f"/api/repositories/{repo_id}/members/{{member_login}}/commits?use_ai=true",
            "batch_analysis": f"/api/repositories/{repo_id}/ai/analyze-batch",
            "developer_insights": f"/api/repositories/{repo_id}/ai/developer-insights"
        }
    }

@router.post("/{repo_id}/ai/analyze-batch")
async def analyze_commits_batch(
    repo_id: int,
    commit_messages: List[str],
    db: Session = Depends(get_db)
):
    """Batch analysis cho nhiều commit messages"""
    try:
        service = MemberAnalysisService(db)
        results = await service.ai_service.analyze_commits_batch(commit_messages)
        
        return {
            "success": True,
            "repository_id": repo_id,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in batch analysis: {str(e)}")

@router.get("/{repo_id}/ai/developer-insights")
async def get_developer_insights(
    repo_id: int,
    db: Session = Depends(get_db)
):
    """Lấy insights về tất cả developers trong repo"""
    try:
        service = MemberAnalysisService(db)
        
        # Get all members
        members = service.get_repository_members(repo_id)
        
        # Get commits for each member and analyze
        developer_commits = {}
        for member in members[:5]:  # Limit to first 5 for demo
            member_login = member['github_username']
            commits_data = service._get_member_commits_raw(repo_id, member_login, 20)
            if commits_data:
                developer_commits[member_login] = [row[2] for row in commits_data]  # messages
        
        # Analyze patterns
        insights = await service.ai_service.analyze_developer_patterns(developer_commits)
        
        return {
            "success": True,
            "repository_id": repo_id,
            "insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting developer insights: {str(e)}")

@router.get("/{repo_id}/ai/model-status")
async def get_ai_model_status(repo_id: int):
    """Kiểm tra trạng thái AI model"""
    try:
        from services.han_ai_service import HANAIService
        ai_service = HANAIService()
        
        return {
            "success": True,
            "repository_id": repo_id,
            "model_loaded": ai_service.is_model_loaded,
            "model_info": {
                "type": "HAN (Hierarchical Attention Network)",
                "purpose": "Commit message analysis and classification",
                "features": [
                    "Semantic understanding",
                    "Multi-level attention",
                    "Context-aware classification"
                ]
            }
        }
    except Exception as e:
        return {
            "success": False,
            "repository_id": repo_id,
            "model_loaded": False,
            "error": str(e)
        }

@router.get("/{repo_id}/branches")
async def get_repository_branches(
    repo_id: int,
    db: Session = Depends(get_db)
):
    """Lấy danh sách branches của repository"""
    try:
        service = MemberAnalysisService(db)
        branches = service.get_repository_branches(repo_id)
        
        return {
            "success": True,
            "repository_id": repo_id,
            "branches": branches,
            "total": len(branches)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching branches: {str(e)}")
