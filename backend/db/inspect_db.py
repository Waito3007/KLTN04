# Script để kiểm tra cấu trúc database và tạo models
import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect

# Add the backend directory to Python path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
print(f"Database URL: {DATABASE_URL}")

if DATABASE_URL:
    try:
        # Sử dụng psycopg2 để kết nối
        if "asyncpg" in DATABASE_URL:
            DATABASE_URL = DATABASE_URL.replace("asyncpg", "psycopg2")
        
        engine = create_engine(DATABASE_URL)
        inspector = inspect(engine)
        
        # Lấy danh sách các bảng
        tables = inspector.get_table_names()
        print(f"\nCác bảng hiện có trong database:")
        print("="*50)
        
        for table in tables:
            print(f"\nBảng: {table}")
            print("-" * 30)
            
            # Lấy thông tin các cột
            columns = inspector.get_columns(table)
            for col in columns:
                nullable = "NULL" if col["nullable"] else "NOT NULL"
                default = f", default={col.get('default')}" if col.get('default') else ""
                print(f"  {col['name']}: {col['type']} {nullable}{default}")
            
            # Lấy thông tin primary keys
            pk_constraint = inspector.get_pk_constraint(table)
            if pk_constraint['constrained_columns']:
                print(f"  Primary Key: {pk_constraint['constrained_columns']}")
            
            # Lấy thông tin foreign keys
            foreign_keys = inspector.get_foreign_keys(table)
            if foreign_keys:
                print("  Foreign Keys:")
                for fk in foreign_keys:
                    print(f"    {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")
            
            print()
        
    except Exception as e:
        print(f"Lỗi khi kết nối database: {e}")
else:
    print("Không tìm thấy DATABASE_URL trong file .env")
