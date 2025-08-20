"""
Script để tạo tất cả các bảng trong cơ sở dữ liệu
Chạy script này khi bạn có database trống hoàn toàn
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from db.database import engine
from db.metadata import metadata
import logging

logger = logging.getLogger(__name__)

def create_all_tables():
    """Tạo tất cả các bảng từ các model đã định nghĩa"""
    try:
        logger.info("🔄 Bắt đầu tạo tất cả các bảng...")
        
        # Import all models để đảm bảo chúng được đăng ký với metadata
        from db.models import (
            users, repositories, commits, branches, issues, 
            pull_requests, collaborators, repository_collaborators,
            user_repositories, assignments, sync_event, project_tasks
        )
        
        # Tạo tất cả các bảng
        metadata.create_all(engine)
        
        logger.info("✅ Đã tạo thành công tất cả các bảng!")
        print("✅ Database tables created successfully!")
        
        # Hiển thị danh sách các bảng đã tạo
        print("\nCác bảng đã được tạo:")
        for table_name in metadata.tables.keys():
            print(f"  - {table_name}")
            
    except Exception as e:
        logger.error(f"❌ Lỗi khi tạo bảng: {e}")
        print(f"❌ Error creating tables: {e}")
        raise e

def check_tables_exist():
    """Kiểm tra xem các bảng đã tồn tại hay chưa"""
    try:
        with engine.connect() as conn:
            # Lấy danh sách bảng hiện có
            result = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in result]
            
        print(f"\nCác bảng hiện có trong database: {len(existing_tables)} bảng")
        for table in existing_tables:
            print(f"  - {table}")
            
        return existing_tables
        
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra bảng: {e}")
        return []

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🗄️  Database Setup Script")
    print("=" * 40)
    
    # Kiểm tra trạng thái hiện tại
    print("1. Kiểm tra bảng hiện có...")
    existing_tables = check_tables_exist()
    
    if existing_tables:
        response = input("\n⚠️  Database đã có bảng. Bạn có muốn tiếp tục tạo bảng mới? (y/N): ")
        if response.lower() != 'y':
            print("🛑 Đã hủy thao tác.")
            sys.exit(0)
    
    # Tạo bảng
    print("\n2. Tạo tất cả các bảng...")
    create_all_tables()
    
    # Kiểm tra lại
    print("\n3. Kiểm tra kết quả...")
    check_tables_exist()
    
    print("\n🎉 Hoàn thành!")
