# backend/db/database.py
import os
from databases import Database
from sqlalchemy import create_engine
from dotenv import load_dotenv
from db.metadata import metadata  # Import metadata từ metadata.py
from db.models.commits import commits
from db.models.repositories import repositories
from db.models.users import users
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

# Kiểm tra kết nối cơ sở dữ liệu
try:
    with sync_engine.connect() as connection:
        print("Kết nối cơ sở dữ liệu thành công!")
except Exception as e:
    print(f"Lỗi kết nối cơ sở dữ liệu: {e}")

# Tạo các bảng
metadata.create_all(sync_engine)
