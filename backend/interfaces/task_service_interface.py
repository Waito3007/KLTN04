from abc import ABC, abstractmethod
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from schemas.task import TaskCreate, TaskUpdate, TaskFilter, TaskResponse, TaskListResponse


class ITaskService(ABC):
    """Interface cho Task Service"""

    @abstractmethod
    async def create_task(
        self, 
        task_data: TaskCreate, 
        created_by_user_id: int,
        created_by_username: str,
        db: AsyncSession
    ) -> TaskResponse:
        """
        Tạo task mới với validation và business rules
        
        Args:
            task_data: Dữ liệu task cần tạo
            created_by_user_id: ID của user tạo task
            created_by_username: Username của user tạo task
            db: Database session
            
        Returns:
            TaskResponse: Task đã tạo
            
        Raises:
            TaskValidationError: Khi dữ liệu không hợp lệ
            TaskLimitExceededError: Khi vượt quá giới hạn task
            TaskRepositoryError: Khi repository không tồn tại
        """
        pass

    @abstractmethod
    async def get_task_by_id(self, task_id: int, db: AsyncSession) -> TaskResponse:
        """
        Lấy task theo ID
        
        Args:
            task_id: ID của task
            db: Database session
            
        Returns:
            TaskResponse: Task tìm được
            
        Raises:
            TaskNotFoundError: Khi không tìm thấy task
        """
        pass

    @abstractmethod
    async def update_task(
        self,
        task_id: int,
        task_data: TaskUpdate,
        user_id: int,
        db: AsyncSession
    ) -> TaskResponse:
        """
        Cập nhật task với business rules và validation
        
        Args:
            task_id: ID của task cần cập nhật
            task_data: Dữ liệu cập nhật
            user_id: ID của user thực hiện cập nhật
            db: Database session
            
        Returns:
            TaskResponse: Task đã cập nhật
            
        Raises:
            TaskNotFoundError: Khi không tìm thấy task
            TaskPermissionError: Khi không có quyền cập nhật
            TaskStatusTransitionError: Khi transition không hợp lệ
        """
        pass

    @abstractmethod
    async def delete_task(self, task_id: int, user_id: int, db: AsyncSession) -> bool:
        """
        Xóa task (soft delete)
        
        Args:
            task_id: ID của task cần xóa
            user_id: ID của user thực hiện xóa
            db: Database session
            
        Returns:
            bool: True nếu xóa thành công
            
        Raises:
            TaskNotFoundError: Khi không tìm thấy task
            TaskPermissionError: Khi không có quyền xóa
        """
        pass

    @abstractmethod
    async def list_tasks(
        self,
        filters: TaskFilter,
        db: AsyncSession,
        page: int = 1,
        page_size: int = 20
    ) -> TaskListResponse:
        """
        Lấy danh sách task với filter và pagination
        
        Args:
            filters: Bộ lọc task
            db: Database session
            page: Số trang (bắt đầu từ 1)
            page_size: Kích thước trang
            
        Returns:
            TaskListResponse: Danh sách task với metadata
        """
        pass
