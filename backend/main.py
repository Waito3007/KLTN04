# backend/main.py

from fastapi import FastAPI
from core.lifespan import lifespan
from core.config import setup_middlewares, setup_routers
from core.logger import setup_logger

setup_logger()  # 👈 Bật logger trước khi chạy app

app = FastAPI(lifespan=lifespan)

setup_middlewares(app)
setup_routers(app)
