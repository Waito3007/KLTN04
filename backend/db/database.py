# backend/db/database.py

from databases import Database
from sqlalchemy import create_engine, MetaData
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Async Database
database = Database(DATABASE_URL)

# Dùng metadata cho model
metadata = MetaData()

# Dành cho migrate nếu có dùng Alembic
engine = create_engine(
    DATABASE_URL.replace("asyncpg", "psycopg2")
)
