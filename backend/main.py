from fastapi import FastAPI
from core.lifespan import lifespan
from core.config import setup_middlewares
from core.logger import setup_logger
import os

from api.routes.auth import auth_router
from api.routes.github import github_router
from api.routes.projects import router as projects_router
from api.routes.sync import sync_router
from api.routes.contributors import router as contributors_router
from api.routes.member_analysis import router as member_analysis_router
from api.routes.repositories import router as repositories_router

setup_logger()  # Bật logger trước khi chạy app

app = FastAPI(lifespan=lifespan)

setup_middlewares(app)

# Include routers
app.include_router(auth_router, prefix="/api")
app.include_router(github_router, prefix="/api")
app.include_router(projects_router, prefix="/api")
app.include_router(sync_router, prefix="/api")
app.include_router(contributors_router, prefix="/api/contributors")
app.include_router(member_analysis_router)
app.include_router(repositories_router)

@app.get("/")
def root():
    return {"message": "TaskFlowAI backend is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000)) 
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Lắng nghe trên tất cả interfaces
        port=port,
        reload=False,  # Tắt reload trên production
        workers=1  
    )