# backend/api/routes/github.py
# File tổng hợp các router GitHub APIs

from fastapi import APIRouter
from .repo import repo_router
from .commit import commit_router
from .branch import branch_router
from .issue import issue_router
from .sync import sync_router

# Router chính cho GitHub APIs
github_router = APIRouter()

# Include các sub-routers
github_router.include_router(repo_router, tags=["repositories"])
github_router.include_router(commit_router, tags=["commits"])
github_router.include_router(branch_router, tags=["branches"])
github_router.include_router(issue_router, tags=["issues"])
github_router.include_router(sync_router, tags=["synchronization"])