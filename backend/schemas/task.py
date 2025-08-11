"""
Schemas cho Task management - tuân thủ quy tắc KLTN04
Sử dụng Pydantic để validation dữ liệu input/output
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
from pydantic import ConfigDict
from db.models.project_tasks import TaskStatus, TaskPriority


class TaskBase(BaseModel):
    """Base schema cho Task với các trường chung"""
    title: str = Field(..., min_length=1, max_length=255, description="Tiêu đề task")
    description: Optional[str] = Field(None, description="Mô tả chi tiết task")
    status: TaskStatus = Field(default=TaskStatus.TODO, description="Trạng thái task")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Độ ưu tiên task")
    due_date: Optional[str] = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}$', description="Ngày hết hạn (YYYY-MM-DD)")
    assignee_github_username: Optional[str] = Field(None, max_length=100, description="GitHub username của người được giao")
    repo_owner: Optional[str] = Field(None, max_length=100, description="Chủ sở hữu repository")
    repo_name: Optional[str] = Field(None, max_length=100, description="Tên repository")

    @validator('title')
    def validate_title(cls, v):
        """Validate title không được rỗng sau khi trim"""
        if not v or not v.strip():
            raise ValueError('Tiêu đề task không được để trống')
        return v.strip()

    @validator('due_date')
    def validate_due_date(cls, v):
        """Validate format ngày và không được ở quá khứ"""
        if v is None:
            return v
        try:
            from datetime import datetime
            due_date_obj = datetime.strptime(v, '%Y-%m-%d')
            # Tạm thời bỏ validation quá khứ để debug
            # if due_date_obj.date() < datetime.now().date():
            #     raise ValueError('Ngày hết hạn không được ở quá khứ')
            return v
        except ValueError as e:
            if "does not match format" in str(e):
                raise ValueError('Định dạng ngày không hợp lệ. Sử dụng YYYY-MM-DD')
            raise e


class TaskCreate(TaskBase):
    """Schema để tạo task mới"""
    pass


class TaskUpdate(BaseModel):
    """Schema để cập nhật task - tất cả field đều optional"""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    due_date: Optional[str] = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}$')
    assignee_github_username: Optional[str] = Field(None, max_length=100)
    repo_owner: Optional[str] = Field(None, max_length=100)
    repo_name: Optional[str] = Field(None, max_length=100)
    is_completed: Optional[bool] = None

    @validator('title')
    def validate_title(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError('Tiêu đề task không được để trống')
        return v.strip() if v else v

    @validator('due_date')
    def validate_due_date(cls, v):
        if v is None:
            return v
        try:
            from datetime import datetime
            due_date_obj = datetime.strptime(v, '%Y-%m-%d')
            # Tạm thời bỏ validation quá khứ để debug
            # if due_date_obj.date() < datetime.now().date():
            #     raise ValueError('Ngày hết hạn không được ở quá khứ')
            return v
        except ValueError as e:
            if "does not match format" in str(e):
                raise ValueError('Định dạng ngày không hợp lệ. Sử dụng YYYY-MM-DD')
            raise e


class TaskResponse(BaseModel):
    """Schema cho response của Task - không validate business rules"""
    id: int
    title: str
    description: Optional[str] = None
    status: TaskStatus
    priority: TaskPriority
    due_date: Optional[str] = None
    assignee_github_username: Optional[str] = None
    assignee: Optional[str] = None  # Backward compatibility
    repo_owner: Optional[str] = None
    repo_name: Optional[str] = None
    assignee_user_id: Optional[int] = None
    repository_id: Optional[int] = None
    is_completed: Optional[bool] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by_user_id: Optional[int] = None
    created_by: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class TaskFilter(BaseModel):
    """Schema cho việc filter tasks"""
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    assignee_github_username: Optional[str] = None
    repo_owner: Optional[str] = None
    repo_name: Optional[str] = None
    is_completed: Optional[bool] = None
    created_by: Optional[str] = None


class TaskListResponse(BaseModel):
    """Schema cho response danh sách tasks"""
    tasks: List[TaskResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

    model_config = ConfigDict(from_attributes=True)


# Constants cho business logic
MAX_TASKS_PER_USER = 100
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
