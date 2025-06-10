from fastapi import FastAPI
from contextlib import asynccontextmanager
from db.database import database
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await database.connect()
        logger.info("✅ Đã kết nối tới database thành công.")
        yield  # Chỉ yield nếu connect thành công
    except Exception as e:
        logger.error(f"❌ Kết nối database thất bại: {e}")
        raise e  # Dừng app nếu không kết nối được DB
    finally:
        try:
            await database.disconnect()
            logger.info("🛑 Đã ngắt kết nối database.")
        except Exception as e:
            logger.error(f"❌ Lỗi khi ngắt kết nối database: {e}")
