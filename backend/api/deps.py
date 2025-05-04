# filepath: c:\SAN\KLTN\KLTN04\backend\api\deps.py
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import database

async def get_db() -> AsyncSession:
    async with database.session() as session:
        yield session