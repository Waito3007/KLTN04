"""
Task Service - Logic nghiệp vụ cho Task management
Tuân thủ nguyên tắc Defensive Programming và Immutability của KLTN04
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, update, delete
from sqlalchemy.exc import SQLAlchemyError

from db.models.project_tasks import project_tasks, TaskStatus, TaskPriority
from db.models.repositories import repositories
from schemas.task import TaskCreate, TaskUpdate, TaskFilter, TaskResponse, TaskListResponse
from core.task_exceptions import (
    TaskNotFoundError, TaskValidationError, TaskPermissionError,
    TaskLimitExceededError, TaskStatusTransitionError, TaskRepositoryError,
    TaskDatabaseError
)

# Constants
MAX_TASKS_PER_USER = 100
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Valid status transitions
VALID_STATUS_TRANSITIONS = {
    TaskStatus.TODO: [TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED],
    TaskStatus.IN_PROGRESS: [TaskStatus.DONE, TaskStatus.TODO, TaskStatus.CANCELLED],
    TaskStatus.DONE: [TaskStatus.TODO, TaskStatus.IN_PROGRESS],  # Allow reopening
    TaskStatus.CANCELLED: [TaskStatus.TODO, TaskStatus.IN_PROGRESS]  # Allow reactivating
}

logger = logging.getLogger(__name__)


class TaskService:
    """
    Service class chứa tất cả logic nghiệp vụ cho Task management
    Tuân thủ nguyên tắc Single Responsibility và Defensive Programming
    """

    def __init__(self):
        """Initialize TaskService"""
        self.logger = logger

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
            TaskResponse: Thông tin task vừa tạo
            
        Raises:
            TaskValidationError: Khi dữ liệu không hợp lệ
            TaskLimitExceededError: Khi vượt quá giới hạn
            TaskRepositoryError: Khi repository không tồn tại
            TaskDatabaseError: Khi có lỗi database
        """
        try:
            # Defensive Programming: Validate đầu vào
            if not task_data:
                raise TaskValidationError("Dữ liệu task không được để trống")
            
            if not created_by_user_id or created_by_user_id <= 0:
                raise TaskValidationError("User ID không hợp lệ", "created_by_user_id", created_by_user_id)
                
            if not created_by_username or not created_by_username.strip():
                raise TaskValidationError("Username không được để trống", "created_by_username", created_by_username)

            # Kiểm tra giới hạn số lượng task
            await self._check_task_limit(created_by_user_id, db)

            # Validate repository nếu được cung cấp
            repository_id = None
            if task_data.repo_owner and task_data.repo_name:
                repository_id = await self._validate_repository(task_data.repo_owner, task_data.repo_name, db)

            # Validate assignee user nếu được cung cấp
            assignee_user_id = None
            if task_data.assignee_github_username:
                assignee_user_id = await self._validate_assignee(task_data.assignee_github_username, db)

            # Tạo task data với validation
            current_time = datetime.utcnow()
            task_dict = {
                "title": task_data.title.strip(),
                "description": task_data.description.strip() if task_data.description else None,
                "assignee_user_id": assignee_user_id,
                "assignee_github_username": task_data.assignee_github_username,
                "status": task_data.status.value,
                "priority": task_data.priority.value,
                "due_date": task_data.due_date,
                "repository_id": repository_id,
                "repo_owner": task_data.repo_owner,
                "repo_name": task_data.repo_name,
                "is_completed": task_data.status == TaskStatus.DONE,
                "created_by_user_id": created_by_user_id,
                "created_by": created_by_username.strip(),
                "created_at": current_time,
                "updated_at": current_time
            }

            # Insert vào database trong transaction
            async with db.begin():
                insert_stmt = project_tasks.insert().values(**task_dict)
                result = await db.execute(insert_stmt)
                task_id = result.inserted_primary_key[0]

                # Fetch task vừa tạo để return
                select_stmt = select(project_tasks).where(project_tasks.c.id == task_id)
                task_row = (await db.execute(select_stmt)).first()

                if not task_row:
                    raise TaskDatabaseError("Không thể tạo task", "create_task")

            self.logger.info(f"Task {task_id} được tạo thành công bởi user {created_by_user_id}")
            return TaskResponse(**task_row._asdict())

        except (TaskValidationError, TaskLimitExceededError, TaskRepositoryError) as e:
            self.logger.warning(f"Validation error khi tạo task: {e}")
            raise
        except SQLAlchemyError as e:
            self.logger.error(f"Database error khi tạo task: {e}")
            raise TaskDatabaseError(str(e), "create_task")
        except Exception as e:
            self.logger.error(f"Unexpected error khi tạo task: {e}")
            raise TaskDatabaseError(f"Lỗi không mong muốn: {e}", "create_task")

    async def get_task_by_id(self, task_id: int, db: AsyncSession) -> TaskResponse:
        """
        Lấy thông tin task theo ID
        
        Args:
            task_id: ID của task
            db: Database session
            
        Returns:
            TaskResponse: Thông tin task
            
        Raises:
            TaskNotFoundError: Khi không tìm thấy task
            TaskValidationError: Khi task_id không hợp lệ
        """
        try:
            # Defensive Programming: Validate đầu vào
            if not task_id or task_id <= 0:
                raise TaskValidationError("Task ID không hợp lệ", "task_id", task_id)

            select_stmt = select(project_tasks).where(project_tasks.c.id == task_id)
            result = await db.execute(select_stmt)
            task_row = result.first()

            if not task_row:
                raise TaskNotFoundError(task_id)

            return TaskResponse(**task_row._asdict())

        except TaskNotFoundError:
            raise
        except SQLAlchemyError as e:
            self.logger.error(f"Database error khi lấy task {task_id}: {e}")
            raise TaskDatabaseError(str(e), "get_task_by_id")

    async def update_task(
        self, 
        task_id: int, 
        task_data: TaskUpdate, 
        user_id: int, 
        db: AsyncSession
    ) -> TaskResponse:
        """
        Cập nhật thông tin task
        
        Args:
            task_id: ID của task cần cập nhật
            task_data: Dữ liệu cần cập nhật
            user_id: ID của user thực hiện cập nhật
            db: Database session
            
        Returns:
            TaskResponse: Thông tin task sau khi cập nhật
            
        Raises:
            TaskNotFoundError: Khi không tìm thấy task
            TaskPermissionError: Khi không có quyền cập nhật
            TaskValidationError: Khi dữ liệu không hợp lệ
            TaskStatusTransitionError: Khi chuyển trạng thái không hợp lệ
        """
        try:
            # Validate đầu vào
            if not task_id or task_id <= 0:
                raise TaskValidationError("Task ID không hợp lệ", "task_id", task_id)
            
            if not user_id or user_id <= 0:
                raise TaskValidationError("User ID không hợp lệ", "user_id", user_id)

            # Lấy task hiện tại
            current_task = await self.get_task_by_id(task_id, db)

            # Kiểm tra quyền cập nhật (chỉ người tạo hoặc người được assign mới được update)
            if not await self._can_update_task(current_task, user_id, db):
                raise TaskPermissionError(task_id, user_id, "cập nhật")

            # Xây dựng dict cập nhật (chỉ các field không None)
            update_dict = {}
            
            if task_data.title is not None:
                update_dict["title"] = task_data.title.strip()
            
            if task_data.description is not None:
                update_dict["description"] = task_data.description.strip() if task_data.description else None
                
            if task_data.status is not None:
                # Validate status transition
                if not await self._is_valid_status_transition(current_task.status, task_data.status):
                    raise TaskStatusTransitionError(task_id, current_task.status, task_data.status.value)
                update_dict["status"] = task_data.status.value
                update_dict["is_completed"] = task_data.status == TaskStatus.DONE
                
            if task_data.priority is not None:
                update_dict["priority"] = task_data.priority.value
                
            if task_data.due_date is not None:
                update_dict["due_date"] = task_data.due_date
                
            if task_data.assignee_github_username is not None:
                if task_data.assignee_github_username:
                    assignee_user_id = await self._validate_assignee(task_data.assignee_github_username, db)
                    update_dict["assignee_user_id"] = assignee_user_id
                    update_dict["assignee_github_username"] = task_data.assignee_github_username
                else:
                    update_dict["assignee_user_id"] = None
                    update_dict["assignee_github_username"] = None
                    
            if task_data.repo_owner is not None or task_data.repo_name is not None:
                if task_data.repo_owner and task_data.repo_name:
                    repository_id = await self._validate_repository(task_data.repo_owner, task_data.repo_name, db)
                    update_dict["repository_id"] = repository_id
                    update_dict["repo_owner"] = task_data.repo_owner
                    update_dict["repo_name"] = task_data.repo_name
                else:
                    update_dict["repository_id"] = None
                    update_dict["repo_owner"] = None
                    update_dict["repo_name"] = None

            if task_data.is_completed is not None:
                update_dict["is_completed"] = task_data.is_completed

            # Nếu không có gì để cập nhật
            if not update_dict:
                return current_task

            update_dict["updated_at"] = datetime.utcnow()

            # Thực hiện update trong transaction
            async with db.begin():
                update_stmt = (
                    update(project_tasks)
                    .where(project_tasks.c.id == task_id)
                    .values(**update_dict)
                )
                await db.execute(update_stmt)

                # Fetch task sau khi update
                updated_task = await self.get_task_by_id(task_id, db)

            self.logger.info(f"Task {task_id} được cập nhật bởi user {user_id}")
            return updated_task

        except (TaskNotFoundError, TaskPermissionError, TaskValidationError, TaskStatusTransitionError) as e:
            self.logger.warning(f"Error khi cập nhật task {task_id}: {e}")
            raise
        except SQLAlchemyError as e:
            self.logger.error(f"Database error khi cập nhật task {task_id}: {e}")
            raise TaskDatabaseError(str(e), "update_task")

    async def delete_task(self, task_id: int, user_id: int, db: AsyncSession) -> bool:
        """
        Xóa task (chỉ người tạo mới được xóa)
        
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
        try:
            # Validate đầu vào
            if not task_id or task_id <= 0:
                raise TaskValidationError("Task ID không hợp lệ", "task_id", task_id)
            
            if not user_id or user_id <= 0:
                raise TaskValidationError("User ID không hợp lệ", "user_id", user_id)

            # Lấy task hiện tại và kiểm tra quyền
            current_task = await self.get_task_by_id(task_id, db)
            
            # Chỉ người tạo mới được xóa
            if current_task.created_by_user_id != user_id:
                raise TaskPermissionError(task_id, user_id, "xóa")

            # Thực hiện xóa trong transaction
            async with db.begin():
                delete_stmt = delete(project_tasks).where(project_tasks.c.id == task_id)
                result = await db.execute(delete_stmt)
                
                if result.rowcount == 0:
                    raise TaskNotFoundError(task_id)

            self.logger.info(f"Task {task_id} được xóa bởi user {user_id}")
            return True

        except (TaskNotFoundError, TaskPermissionError, TaskValidationError) as e:
            self.logger.warning(f"Error khi xóa task {task_id}: {e}")
            raise

        except SQLAlchemyError as e:
            self.logger.error(f"Database error khi xóa task {task_id}: {e}")
            raise TaskDatabaseError(str(e), "delete_task")
            return True

        except (TaskNotFoundError, TaskPermissionError, TaskValidationError) as e:
            self.logger.warning(f"Error khi xóa task {task_id}: {e}")
            raise
        except SQLAlchemyError as e:
            self.logger.error(f"Database error khi xóa task {task_id}: {e}")
            raise TaskDatabaseError(str(e), "delete_task")

    async def list_tasks(
        self, 
        filters: Optional[TaskFilter] = None,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
        user_id: Optional[int] = None,
        db: AsyncSession = None
    ) -> TaskListResponse:
        """
        Lấy danh sách tasks với filter và pagination
        
        Args:
            filters: Bộ lọc tasks
            page: Số trang (bắt đầu từ 1)
            page_size: Số lượng items per page
            user_id: ID user (để filter tasks liên quan đến user)
            db: Database session
            
        Returns:
            TaskListResponse: Danh sách tasks với pagination info
        """
        try:
            # Validate pagination
            if page < 1:
                page = 1
            if page_size < 1 or page_size > MAX_PAGE_SIZE:
                page_size = DEFAULT_PAGE_SIZE

            # Xây dựng query conditions
            conditions = []
            
            if filters:
                if filters.status:
                    conditions.append(project_tasks.c.status == filters.status.value)
                if filters.priority:
                    conditions.append(project_tasks.c.priority == filters.priority.value)
                if filters.assignee_github_username:
                    conditions.append(project_tasks.c.assignee_github_username == filters.assignee_github_username)
                if filters.repo_owner:
                    conditions.append(project_tasks.c.repo_owner == filters.repo_owner)
                if filters.repo_name:
                    conditions.append(project_tasks.c.repo_name == filters.repo_name)
                if filters.is_completed is not None:
                    conditions.append(project_tasks.c.is_completed == filters.is_completed)
                if filters.created_by:
                    conditions.append(project_tasks.c.created_by == filters.created_by)

            # Filter theo user_id nếu được cung cấp (my_tasks)
            if user_id:
                user_conditions = or_(
                    project_tasks.c.created_by_user_id == user_id,
                    project_tasks.c.assignee_user_id == user_id
                )
                conditions.append(user_conditions)

            # Xây dựng query
            where_clause = and_(*conditions) if conditions else None

            # Count total items
            count_query = select(func.count(project_tasks.c.id))
            if where_clause is not None:
                count_query = count_query.where(where_clause)
            
            count_result = await db.execute(count_query)
            total = count_result.scalar()

            # Calculate pagination
            total_pages = (total + page_size - 1) // page_size if total > 0 else 0
            offset = (page - 1) * page_size

            # Query tasks with pagination
            select_query = select(project_tasks).order_by(project_tasks.c.created_at.desc())
            if where_clause is not None:
                select_query = select_query.where(where_clause)
            select_query = select_query.offset(offset).limit(page_size)

            result = await db.execute(select_query)
            task_rows = result.fetchall()

            # Convert to TaskResponse objects
            tasks = [TaskResponse(**row._asdict()) for row in task_rows]

            return TaskListResponse(
                tasks=tasks,
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages
            )

        except SQLAlchemyError as e:
            self.logger.error(f"Database error khi list tasks: {e}")
            raise TaskDatabaseError(str(e), "list_tasks")

    # Private helper methods
    async def _check_task_limit(self, user_id: int, db: AsyncSession) -> None:
        """Kiểm tra giới hạn số lượng task của user"""
        count_query = select(func.count(project_tasks.c.id)).where(
            project_tasks.c.created_by_user_id == user_id
        )
        current_count = (await db.execute(count_query)).scalar()
        
        if current_count >= MAX_TASKS_PER_USER:
            raise TaskLimitExceededError(user_id, current_count, MAX_TASKS_PER_USER)

    async def _validate_repository(self, repo_owner: str, repo_name: str, db: AsyncSession) -> Optional[int]:
        """Validate repository exists và return repository_id"""
        # TODO: Implement repository validation
        # Hiện tại return None, cần implement sau khi có repository service
        return None

    async def _validate_assignee(self, github_username: str, db: AsyncSession) -> Optional[int]:
        """Validate assignee user exists và return user_id"""
        # TODO: Implement user validation
        # Hiện tại return None, cần implement sau khi có user service
        return None

    async def _can_update_task(self, task: TaskResponse, user_id: int, db: AsyncSession) -> bool:
        """Kiểm tra user có quyền update task không"""
        # Người tạo hoặc người được assign có thể update
        return (
            task.created_by_user_id == user_id or 
            task.assignee_user_id == user_id
        )

    async def _is_valid_status_transition(self, from_status: str, to_status: TaskStatus) -> bool:
        """Kiểm tra status transition có hợp lệ không"""
        try:
            from_status_enum = TaskStatus(from_status)
            return to_status in VALID_STATUS_TRANSITIONS.get(from_status_enum, [])
        except ValueError:
            return False
