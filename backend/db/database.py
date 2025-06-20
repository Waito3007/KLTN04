# backend/db/database.py
import os
import sys
from databases import Database
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add the backend directory to Python path to ensure proper imports
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from db.metadata import metadata  # Import metadata từ metadata.py
from db.models.commits import commits
from db.models.repositories import repositories
from db.models.users import users
from db.models.branches import branches
from db.models.collaborators import collaborators
from db.models.project_tasks import project_tasks
from db.models.repository_collaborators import repository_collaborators
from db.models.user_repositories import user_repositories
from db.models.issues import issues
from db.models.pull_requests import pull_requests
from db.models.assignments import assignments

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Async Database
database = Database(DATABASE_URL)

# Engine đồng bộ để tạo bảng
sync_engine = create_engine(
    DATABASE_URL.replace("asyncpg", "psycopg2")  # Thay asyncpg bằng psycopg2
)

# Session maker for SQLAlchemy ORM
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Kiểm tra kết nối cơ sở dữ liệu
try:
    with sync_engine.connect() as connection:
        print("Kết nối cơ sở dữ liệu thành công!")
except Exception as e:
    print(f"Lỗi kết nối cơ sở dữ liệu: {e}")

# Tạo các bảng
metadata.create_all(sync_engine)

engine = sync_engine
