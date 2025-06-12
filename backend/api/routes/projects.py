from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, insert, update, delete, and_
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

# from core.security import get_current_user  # Temporarily disabled
from db.database import get_db, engine
from db.models.project_tasks import project_tasks, TaskStatus, TaskPriority

router = APIRouter()

# Temporary mock user dependency
async def get_current_user():
    return {"username": "test_user", "id": 1}

# Pydantic models cho Task
class TaskBase(BaseModel):
    title: str
    description: Optional[str] = None
    assignee: str
    priority: str = "medium"  # low, medium, high
    status: str = "todo"  # todo, in_progress, done
    due_date: Optional[str] = None

class TaskCreate(TaskBase):
    repo_owner: str
    repo_name: str

class TaskUpdate(TaskBase):
    pass

class TaskResponse(TaskBase):
    id: int
    repo_owner: str
    repo_name: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

@router.get("/projects/{owner}/{repo}/tasks", response_model=List[TaskResponse])
async def get_project_tasks(
    owner: str,
    repo: str,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lấy danh sách tasks của repository"""
    try:
        # Query tasks from database
        with engine.connect() as conn:
            query = select(project_tasks).where(
                and_(
                    project_tasks.c.repo_owner == owner,
                    project_tasks.c.repo_name == repo
                )
            ).order_by(project_tasks.c.created_at.desc())
            
            result = conn.execute(query)
            tasks = []
            
            for row in result:
                task_dict = {
                    "id": row.id,
                    "title": row.title,
                    "description": row.description,
                    "assignee": row.assignee,
                    "priority": row.priority.value if row.priority else "medium",
                    "status": row.status.value if row.status else "todo",
                    "due_date": row.due_date,
                    "repo_owner": row.repo_owner,
                    "repo_name": row.repo_name,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at
                }
                tasks.append(task_dict)
            
            return tasks
    except Exception as e:
        print(f"Database error: {e}")
        # Fallback to empty list if database error
        return []

@router.post("/projects/{owner}/{repo}/tasks", response_model=TaskResponse)
async def create_project_task(
    owner: str,
    repo: str,
    task: TaskCreate,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Tạo task mới cho repository"""
    try:
        # Insert task into database
        with engine.connect() as conn:
            # Validate priority and status
            priority_enum = TaskPriority.MEDIUM
            if task.priority == "low":
                priority_enum = TaskPriority.LOW
            elif task.priority == "high":
                priority_enum = TaskPriority.HIGH
                
            status_enum = TaskStatus.TODO
            if task.status == "in_progress":
                status_enum = TaskStatus.IN_PROGRESS
            elif task.status == "done":
                status_enum = TaskStatus.DONE
            
            insert_stmt = insert(project_tasks).values(
                title=task.title,
                description=task.description,
                assignee=task.assignee,
                priority=priority_enum,
                status=status_enum,
                due_date=task.due_date,
                repo_owner=owner,
                repo_name=repo,
                created_by=current_user["username"]
            )
            
            result = conn.execute(insert_stmt)
            conn.commit()
            
            # Get the created task
            task_id = result.inserted_primary_key[0]
            query = select(project_tasks).where(project_tasks.c.id == task_id)
            created_task = conn.execute(query).fetchone()
            
            return {
                "id": created_task.id,
                "title": created_task.title,
                "description": created_task.description,
                "assignee": created_task.assignee,
                "priority": created_task.priority.value,
                "status": created_task.status.value,
                "due_date": created_task.due_date,
                "repo_owner": created_task.repo_owner,
                "repo_name": created_task.repo_name,
                "created_at": created_task.created_at,
                "updated_at": created_task.updated_at
            }
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.put("/projects/{owner}/{repo}/tasks/{task_id}", response_model=TaskResponse)
async def update_project_task(
    owner: str,
    repo: str,
    task_id: int,
    task_update: TaskUpdate,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cập nhật task"""
    try:
        with engine.connect() as conn:
            # Check if task exists
            check_query = select(project_tasks).where(
                and_(
                    project_tasks.c.id == task_id,
                    project_tasks.c.repo_owner == owner,
                    project_tasks.c.repo_name == repo
                )
            )
            existing_task = conn.execute(check_query).fetchone()
            
            if not existing_task:
                raise HTTPException(status_code=404, detail="Task not found")
            
            # Validate priority and status
            priority_enum = TaskPriority.MEDIUM
            if task_update.priority == "low":
                priority_enum = TaskPriority.LOW
            elif task_update.priority == "high":
                priority_enum = TaskPriority.HIGH
                
            status_enum = TaskStatus.TODO
            if task_update.status == "in_progress":
                status_enum = TaskStatus.IN_PROGRESS
            elif task_update.status == "done":
                status_enum = TaskStatus.DONE
            
            # Update task
            update_stmt = update(project_tasks).where(
                project_tasks.c.id == task_id
            ).values(
                title=task_update.title,
                description=task_update.description,
                assignee=task_update.assignee,
                priority=priority_enum,
                status=status_enum,
                due_date=task_update.due_date
            )
            
            conn.execute(update_stmt)
            conn.commit()
            
            # Get updated task
            updated_task = conn.execute(check_query).fetchone()
            
            return {
                "id": updated_task.id,
                "title": updated_task.title,
                "description": updated_task.description,
                "assignee": updated_task.assignee,
                "priority": updated_task.priority.value,
                "status": updated_task.status.value,
                "due_date": updated_task.due_date,
                "repo_owner": updated_task.repo_owner,
                "repo_name": updated_task.repo_name,
                "created_at": updated_task.created_at,
                "updated_at": updated_task.updated_task
            }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.delete("/projects/{owner}/{repo}/tasks/{task_id}")
async def delete_project_task(
    owner: str,
    repo: str,
    task_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Xóa task"""
    try:
        with engine.connect() as conn:
            # Check if task exists
            check_query = select(project_tasks).where(
                and_(
                    project_tasks.c.id == task_id,
                    project_tasks.c.repo_owner == owner,
                    project_tasks.c.repo_name == repo
                )
            )
            existing_task = conn.execute(check_query).fetchone()
            
            if not existing_task:
                raise HTTPException(status_code=404, detail="Task not found")
            
            # Delete task
            delete_stmt = delete(project_tasks).where(
                project_tasks.c.id == task_id
            )
            
            conn.execute(delete_stmt)
            conn.commit()
            
            return {"message": "Task deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/projects/{owner}/{repo}/collaborators")
async def get_project_collaborators(
    owner: str,
    repo: str,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lấy danh sách collaborators của repository"""
    try:
        # Mock data - trong thực tế sẽ gọi GitHub API
        collaborators = [
            {
                "login": "john_doe",
                "avatar_url": "https://via.placeholder.com/32",
                "type": "User"
            },
            {
                "login": "jane_smith", 
                "avatar_url": "https://via.placeholder.com/32",
                "type": "User"
            },
            {
                "login": owner,
                "avatar_url": "https://via.placeholder.com/32",
                "type": "Owner"
            }
        ]
        return collaborators
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))