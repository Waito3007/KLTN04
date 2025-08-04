"""
Task API Routes - RESTful endpoints cho Task management
Tuân thủ kiến trúc KLTN04: Logic trong service, routes chỉ handle HTTP
"""

from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user_dict, get_task_service
from services.task_service import TaskService
from schemas.task import (
    TaskCreate, TaskUpdate, TaskResponse, TaskListResponse, 
    TaskFilter, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
)
from core.task_exceptions import (
    TaskNotFoundError, TaskValidationError, TaskPermissionError,
    TaskLimitExceededError, TaskStatusTransitionError, TaskRepositoryError,
    TaskDatabaseError
)
from db.models.project_tasks import TaskStatus, TaskPriority

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


@router.get("/health", status_code=200)
async def health_check():
    """Health check endpoint cho Task API - không cần authentication"""
    return {
        "status": "success",
        "message": "Task API is running",
        "service": "Task Management",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    task_data: TaskCreate,
    current_user: dict = Depends(get_current_user_dict),
    task_service: TaskService = Depends(get_task_service),
    db: AsyncSession = Depends(get_db)
):
    """
    Tạo task mới
    
    - **title**: Tiêu đề task (bắt buộc)
    - **description**: Mô tả chi tiết task
    - **status**: Trạng thái task (mặc định: TODO)
    - **priority**: Độ ưu tiên (mặc định: MEDIUM)
    - **due_date**: Ngày hết hạn (format: YYYY-MM-DD)
    - **assignee_github_username**: GitHub username của người được giao
    - **repo_owner**: Chủ sở hữu repository
    - **repo_name**: Tên repository
    """
    try:
        return await task_service.create_task(
            task_data=task_data,
            created_by_user_id=current_user["id"],
            created_by_username=current_user["username"],
            db=db
        )
    except TaskValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": e.message, "error_code": e.error_code, "details": e.details}
        )
    except TaskLimitExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": e.message, "error_code": e.error_code, "details": e.details}
        )
    except TaskRepositoryError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": e.message, "error_code": e.error_code, "details": e.details}
        )
    except TaskDatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": e.message, "error_code": e.error_code}
        )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: int,
    task_service: TaskService = Depends(get_task_service),
    db: AsyncSession = Depends(get_db)
):
    """
    Lấy thông tin task theo ID
    """
    try:
        return await task_service.get_task_by_id(task_id, db)
    except TaskNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": e.message, "error_code": e.error_code, "details": e.details}
        )
    except TaskValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": e.message, "error_code": e.error_code, "details": e.details}
        )
    except TaskDatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": e.message, "error_code": e.error_code}
        )


@router.put("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: int,
    task_data: TaskUpdate,
    current_user: dict = Depends(get_current_user_dict),
    task_service: TaskService = Depends(get_task_service),
    db: AsyncSession = Depends(get_db)
):
    """
    Cập nhật thông tin task
    
    Chỉ người tạo task hoặc người được assign mới có quyền cập nhật
    """
    try:
        return await task_service.update_task(
            task_id=task_id,
            task_data=task_data,
            user_id=current_user["id"],
            db=db
        )
    except TaskNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": e.message, "error_code": e.error_code, "details": e.details}
        )
    except TaskPermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": e.message, "error_code": e.error_code, "details": e.details}
        )
    except TaskValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": e.message, "error_code": e.error_code, "details": e.details}
        )
    except TaskStatusTransitionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": e.message, "error_code": e.error_code, "details": e.details}
        )
    except TaskDatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": e.message, "error_code": e.error_code}
        )


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: int,
    current_user: dict = Depends(get_current_user_dict),
    task_service: TaskService = Depends(get_task_service),
    db: AsyncSession = Depends(get_db)
):
    """
    Xóa task
    
    Chỉ người tạo task mới có quyền xóa
    """
    try:
        await task_service.delete_task(
            task_id=task_id,
            user_id=current_user["id"],
            db=db
        )
    except TaskNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": e.message, "error_code": e.error_code, "details": e.details}
        )
    except TaskPermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": e.message, "error_code": e.error_code, "details": e.details}
        )
    except TaskValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": e.message, "error_code": e.error_code, "details": e.details}
        )
    except TaskDatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": e.message, "error_code": e.error_code}
        )


@router.get("/", response_model=TaskListResponse)
async def list_tasks(
    # Filter parameters
    status: Optional[TaskStatus] = Query(None, description="Lọc theo trạng thái task"),
    priority: Optional[TaskPriority] = Query(None, description="Lọc theo độ ưu tiên"),
    assignee_github_username: Optional[str] = Query(None, description="Lọc theo người được giao"),
    repo_owner: Optional[str] = Query(None, description="Lọc theo chủ repository"),
    repo_name: Optional[str] = Query(None, description="Lọc theo tên repository"),
    is_completed: Optional[bool] = Query(None, description="Lọc theo trạng thái hoàn thành"),
    created_by: Optional[str] = Query(None, description="Lọc theo người tạo"),
    
    # Pagination parameters
    page: int = Query(1, ge=1, description="Số trang (bắt đầu từ 1)"),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Số lượng items per page"),
    
    # Filter by current user's tasks
    my_tasks: bool = Query(False, description="Chỉ lấy tasks liên quan đến user hiện tại"),
    
    current_user: dict = Depends(get_current_user_dict),
    task_service: TaskService = Depends(get_task_service),
    db: AsyncSession = Depends(get_db)
):
    """
    Lấy danh sách tasks với filter và pagination
    
    ### Query Parameters:
    - **status**: Lọc theo trạng thái (TODO, IN_PROGRESS, DONE, CANCELLED)
    - **priority**: Lọc theo độ ưu tiên (LOW, MEDIUM, HIGH, URGENT)
    - **assignee_github_username**: Lọc theo người được giao task
    - **repo_owner**: Lọc theo chủ repository
    - **repo_name**: Lọc theo tên repository
    - **is_completed**: Lọc theo trạng thái hoàn thành
    - **created_by**: Lọc theo người tạo task
    - **page**: Số trang (mặc định: 1)
    - **page_size**: Số lượng items per page (mặc định: 20, tối đa: 100)
    - **my_tasks**: Chỉ lấy tasks liên quan đến user hiện tại (mặc định: false)
    """
    try:
        # Tạo filter object
        filters = TaskFilter(
            status=status,
            priority=priority,
            assignee_github_username=assignee_github_username,
            repo_owner=repo_owner,
            repo_name=repo_name,
            is_completed=is_completed,
            created_by=created_by
        )
        
        # Nếu my_tasks=True thì chỉ lấy tasks liên quan đến user hiện tại
        user_id = current_user["id"] if my_tasks else None
        
        return await task_service.list_tasks(
            filters=filters,
            page=page,
            page_size=page_size,
            user_id=user_id,
            db=db
        )
    except TaskDatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": e.message, "error_code": e.error_code}
        )


# Utility endpoints
@router.get("/stats/summary")
async def get_task_stats(
    current_user: dict = Depends(get_current_user_dict),
    task_service: TaskService = Depends(get_task_service),
    db: AsyncSession = Depends(get_db)
):
    """
    Lấy thống kê tổng quan về tasks của user hiện tại
    """
    try:
        user_id = current_user["id"]
        
        # Lấy tasks theo từng status
        stats = {}
        for task_status in TaskStatus:
            filters = TaskFilter(status=task_status)
            result = await task_service.list_tasks(
                filters=filters,
                page=1,
                page_size=1,  # Chỉ cần count
                user_id=user_id,
                db=db
            )
            stats[task_status.value.lower()] = result.total
        
        # Tính tổng
        stats["total"] = sum(stats.values())
        
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Lỗi khi lấy thống kê: {str(e)}", "error_code": "STATS_ERROR"}
        )


@router.patch("/{task_id}/status", response_model=TaskResponse)
async def update_task_status(
    task_id: int,
    status: TaskStatus,
    current_user: dict = Depends(get_current_user_dict),
    task_service: TaskService = Depends(get_task_service),
    db: AsyncSession = Depends(get_db)
):
    """
    Cập nhật chỉ trạng thái của task (tiện lợi cho UI)
    """
    try:
        task_data = TaskUpdate(status=status)
        return await task_service.update_task(
            task_id=task_id,
            task_data=task_data,
            user_id=current_user["id"],
            db=db
        )
    except TaskNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": e.message, "error_code": e.error_code, "details": e.details}
        )
    except TaskPermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": e.message, "error_code": e.error_code, "details": e.details}
        )
    except TaskStatusTransitionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": e.message, "error_code": e.error_code, "details": e.details}
        )
    except TaskDatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": e.message, "error_code": e.error_code}
        )
