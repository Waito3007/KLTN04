# backend/main.py

from fastapi import FastAPI
from core.lifespan import lifespan
from core.config import setup_middlewares
from core.logger import setup_logger

from api.routes.auth import auth_router
from api.routes.github import github_router
from api.routes.projects import router as projects_router
from api.routes.sync import sync_router
from api.routes.contributors import router as contributors_router
from api.routes.member_analysis import router as member_analysis_router
from api.routes.commit_routes import router as commit_router
import sys
import os

# Add the parent directory to the sys.path to allow imports from the 'ai' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes.repositories import router as repositories_router
#from aiApi.api.endpoints import router as ai_router

setup_logger()  # Bật logger trước khi chạy app

app = FastAPI(lifespan=lifespan)

setup_middlewares(app)

# Include routers trực tiếp
app.include_router(auth_router, prefix="/api")
app.include_router(github_router, prefix="/api")
app.include_router(projects_router, prefix="/api")
app.include_router(sync_router, prefix="/api")
app.include_router(contributors_router, prefix="/api/contributors")
app.include_router(member_analysis_router)  # Already has /api prefix
app.include_router(repositories_router)  # Already has /api prefix
app.include_router(commit_router)  # Already has /api prefix
#app.include_router(ai_router, prefix="/api/ai")

@app.get("/")
def root():
    return {"message": "TaskFlowAI backend is running "}

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
