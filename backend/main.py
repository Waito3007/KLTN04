# backend/main.py
from fastapi import FastAPI
from core.lifespan import lifespan
from core.config import setup_middlewares
from core.logger import setup_logger
from services.ai_service import router as ai_router
from api.routes.commit_routes import router as commit_router

from api.routes.auth import auth_router
from api.routes.github import github_router
from api.routes.projects import router as projects_router
from api.routes.ai import ai_router as han_ai_router

setup_logger()  # Bật logger trước khi chạy app

app = FastAPI(lifespan=lifespan)

setup_middlewares(app)

# Include routers trực tiếp
app.include_router(auth_router, prefix="/api")
app.include_router(github_router, prefix="/api")
app.include_router(projects_router, prefix="/api")
app.include_router(han_ai_router, prefix="/api")
app.include_router(ai_router, prefix="/ai")  # giữ lại nếu cần legacy
app.include_router(commit_router)

@app.get("/")
def root():
    return {"message": "TaskFlowAI backend is running "}
