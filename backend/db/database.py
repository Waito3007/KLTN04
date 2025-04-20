# backend/db/database.py
import os
from databases import Database
from sqlalchemy import create_engine
from dotenv import load_dotenv
from backend.db.metadata import metadata  # Import metadata từ metadata.py
from backend.db.models.commits import commits
from backend.db.models.repositories import repositories
from backend.db.models.users import users
from backend.db.models.pull_requests import pull_requests
from backend.db.models.assignments import assignments

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Async Database
database = Database(DATABASE_URL)

# Engine đồng bộ để tạo bảng
sync_engine = create_engine(
    DATABASE_URL.replace("asyncpg", "psycopg2")  # Thay asyncpg bằng psycopg2
)

# Tạo các bảng
metadata.create_all(sync_engine)
