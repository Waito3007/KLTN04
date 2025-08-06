# backend/main.py

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from core.lifespan import lifespan
from core.config import setup_middlewares
from core.logger import setup_logger
from api.routes.auth import auth_router
from api.routes.github import github_router
from api.routes.projects import router as projects_router
from api.routes.sync import sync_router
from api.routes.repo_manager import repo_manager_router
from api.routes.sync_events import sync_events_router, websocket_sync_events
from api.routes.contributors import router as contributors_router
from api.routes.han_commitanalyst import router as han_commit_analyst_router
from api.routes.multifusion_commitanalyst import router as multifusion_commit_analyst_router
from api.routes.member_analysis import router as member_analysis_router
from api.routes.commit_routes import router as commit_router
from api.routes.area_analysis import area_analysis_router
from api.routes.risk_analysis import risk_analysis_router # New import
from api.routes.skill_profile import skill_profile_router # New import
from api.routes.ai_status import router as ai_status_router # New AI status router
from api.routes.tasks import router as tasks_router # Task management router
from api.routes.dashboard import router as dashboard_router # Dashboard analytics router
import sys
import os

# Add the parent directory to the sys.path to allow imports from the 'ai' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes.repositories import router as repositories_router
#from aiApi.api.endpoints import router as ai_router

setup_logger()  # Bật logger trước khi chạy app

app = FastAPI(lifespan=lifespan)

# PI(lifespan=lifespan)

setup_middlewares(app)

# Include routers trực tiếp
app.include_router(auth_router, prefix="/api")
app.include_router(github_router, prefix="/api")
app.include_router(projects_router, prefix="/api")
app.include_router(sync_router, prefix="/api")
app.include_router(repo_manager_router, prefix="/api")
app.include_router(sync_events_router, prefix="/api/sync-events")



app.include_router(contributors_router, prefix="/api/contributors")
app.include_router(han_commit_analyst_router, prefix="/api/han-commit-analysis")
app.include_router(multifusion_commit_analyst_router, prefix="/api/multifusion-commit-analysis")
app.include_router(member_analysis_router)
app.include_router(repositories_router)  # Already has /api prefix
app.include_router(commit_router)  # Already has /api prefix
app.include_router(area_analysis_router)
app.include_router(risk_analysis_router) # New router
app.include_router(skill_profile_router) # New router
app.include_router(ai_status_router) # New AI status router
app.include_router(tasks_router) # Task management router
app.include_router(dashboard_router, prefix="/api/dashboard") # Dashboard analytics router
#app.include_router(ai_router, prefix="/api/ai")

@app.get("/")
def root():
    return {"message": "TaskFlowAI backend is running "}

@app.get("/api/sync-events/test")
def test_sync_events():
    return {"message": "Sync events endpoint is working", "websocket_url": "/api/sync-events/ws"}

# --- HAN model patch for torch.load (SimpleTokenizer) ---
import os
import types
import sys
try:
    from ai.testmodelAi.han_model_real_test_fixed import SimpleTokenizer
    import torch
    # Patch đúng module path cho pickle
    sys.modules['SimpleTokenizer'] = SimpleTokenizer
    sys.modules['ai.testmodelAi.han_model_real_test_fixed.SimpleTokenizer'] = SimpleTokenizer
    # Nếu cần, đăng ký vào torch.serialization
    if hasattr(torch.serialization, 'add_safe_class'):
        torch.serialization.add_safe_class(SimpleTokenizer)
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals({'SimpleTokenizer': SimpleTokenizer})
except Exception as e:
    print(f"[WARN] Could not patch SimpleTokenizer for torch.load: {e}")
# --- END PATCH ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)