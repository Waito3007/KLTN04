# backend/main.py
from fastapi import FastAPI
from core.lifespan import lifespan
from core.config import setup_middlewares
from core.logger import setup_logger
from core.config import setup_middlewares, setup_routers


from api.routes.auth import auth_router
from api.routes.github import github_router

setup_logger()  # Bật logger trước khi chạy app


app = FastAPI(lifespan=lifespan)

setup_routers(app)

setup_middlewares(app)

# Include routers trực tiếp
app.include_router(auth_router, prefix="/api")
app.include_router(github_router, prefix="/api")
@app.get("/")
def root():
    return {"message": "TaskFlowAI backend is running "}
