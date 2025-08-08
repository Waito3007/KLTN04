"""
Custom Exceptions cho Task Service - tuân thủ quy tắc KLTN04
Tất cả exceptions kế thừa từ base exception class
"""

from typing import Optional, Any, Dict


class TaskBaseException(Exception):
    """Base exception class cho tất cả Task-related exceptions"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class TaskNotFoundError(TaskBaseException):
    """Exception khi không tìm thấy task"""
    
    def __init__(self, task_id: int):
        super().__init__(
            message=f"Không tìm thấy task với ID: {task_id}",
            error_code="TASK_NOT_FOUND",
            details={"task_id": task_id}
        )


class TaskValidationError(TaskBaseException):
    """Exception khi dữ liệu task không hợp lệ"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
            
        super().__init__(
            message=message,
            error_code="TASK_VALIDATION_ERROR",
            details=details
        )


class TaskPermissionError(TaskBaseException):
    """Exception khi user không có quyền thao tác task"""
    
    def __init__(self, task_id: int, user_id: int, action: str):
        super().__init__(
            message=f"User {user_id} không có quyền {action} task {task_id}",
            error_code="TASK_PERMISSION_DENIED",
            details={
                "task_id": task_id,
                "user_id": user_id,
                "action": action
            }
        )


class TaskLimitExceededError(TaskBaseException):
    """Exception khi vượt quá giới hạn số lượng task"""
    
    def __init__(self, user_id: int, current_count: int, max_allowed: int):
        super().__init__(
            message=f"User {user_id} đã vượt quá giới hạn {max_allowed} tasks (hiện tại: {current_count})",
            error_code="TASK_LIMIT_EXCEEDED",
            details={
                "user_id": user_id,
                "current_count": current_count,
                "max_allowed": max_allowed
            }
        )


class TaskStatusTransitionError(TaskBaseException):
    """Exception khi chuyển trạng thái task không hợp lệ"""
    
    def __init__(self, task_id: int, from_status: str, to_status: str):
        super().__init__(
            message=f"Không thể chuyển task {task_id} từ trạng thái '{from_status}' sang '{to_status}'",
            error_code="INVALID_STATUS_TRANSITION",
            details={
                "task_id": task_id,
                "from_status": from_status,
                "to_status": to_status
            }
        )


class TaskRepositoryError(TaskBaseException):
    """Exception khi có lỗi liên quan đến repository"""
    
    def __init__(self, message: str, repo_owner: Optional[str] = None, repo_name: Optional[str] = None):
        details = {}
        if repo_owner:
            details["repo_owner"] = repo_owner
        if repo_name:
            details["repo_name"] = repo_name
            
        super().__init__(
            message=message,
            error_code="TASK_REPOSITORY_ERROR",
            details=details
        )


class TaskDatabaseError(TaskBaseException):
    """Exception khi có lỗi database"""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(
            message=f"Lỗi database khi {operation}: {message}" if operation else f"Lỗi database: {message}",
            error_code="TASK_DATABASE_ERROR",
            details={"operation": operation} if operation else {}
        )
