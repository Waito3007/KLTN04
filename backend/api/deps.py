# KLTN04\backend\api\deps.py
# File chứa các dependencies (phụ thuộc) chung của ứng dụng

# Import AsyncSession từ SQLAlchemy để làm việc với database async
from sqlalchemy.ext.asyncio import AsyncSession

# Import kết nối database từ module database
from db.database import database

# Dependency (phụ thuộc) để lấy database session
async def get_db() -> AsyncSession:
    """
    Dependency tạo và quản lý database session
    
    Cách hoạt động:
    - Tạo một async session mới từ connection pool
    - Yield session để sử dụng trong request
    - Đảm bảo session được đóng sau khi request hoàn thành
    
    Returns:
        AsyncSession: Session database async để tương tác với DB
    """
    # Tạo và quản lý session thông qua context manager
    async with database.session() as session:
        # Yield session để sử dụng trong route
        yield session
        # Session sẽ tự động đóng khi ra khỏi block with