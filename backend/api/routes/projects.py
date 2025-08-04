from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, insert, update, delete, and_, func, or_
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

# from core.security import get_current_user  # Temporarily disabled
from core.security import get_current_user, CurrentUser
from db.database import get_db, engine
from db.models.project_tasks import project_tasks, TaskStatus, TaskPriority

router = APIRouter()

# Temporary mock user dependency - REMOVED, using real auth now
# async def get_current_user():
#     return {"username": "test_user", "id": 1}

# Pydantic models cho Task
class TaskBase(BaseModel):
    title: str
    description: Optional[str] = None
    assignee: str
    priority: str = "MEDIUM"  # LOW, MEDIUM, HIGH, URGENT
    status: str = "TODO"  # TODO, IN_PROGRESS, DONE, CANCELLED
    due_date: Optional[str] = None

class TaskCreate(TaskBase):
    # repo_owner và repo_name sẽ được lấy từ URL path, không cần trong request body
    pass

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
    current_user: CurrentUser = Depends(get_current_user),
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
                )            ).order_by(project_tasks.c.created_at.desc())
            
            result = conn.execute(query)
            tasks = []
            
            for row in result:
                task_dict = {
                    "id": row.id,
                    "title": row.title,
                    "description": row.description,
                    "assignee": row.assignee_github_username,  # Use correct field name
                    "priority": row.priority if row.priority else "MEDIUM",  # Already string
                    "status": row.status if row.status else "TODO",  # Already string
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
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Tạo task mới cho repository"""
    try:
        # Debug logging
        print(f"🔍 DEBUG: Received task data: {task.dict()}")
        print(f"🔍 DEBUG: Task assignee: {getattr(task, 'assignee', None)}")
        print(f"🔍 DEBUG: Task assignee_github_username: {getattr(task, 'assignee_github_username', None)}")
        
        # Insert task into database
        with engine.connect() as conn:            # Validate priority and status
            priority_enum = TaskPriority.MEDIUM
            if task.priority == "LOW":
                priority_enum = TaskPriority.LOW            
            elif task.priority == "HIGH":
                priority_enum = TaskPriority.HIGH
            
            status_enum = TaskStatus.TODO
            if task.status == "IN_PROGRESS":
                status_enum = TaskStatus.IN_PROGRESS
            elif task.status == "DONE":
                status_enum = TaskStatus.DONE
              # Handle due_date conversion
            due_date_value = None
            if task.due_date:
                try:
                    from datetime import datetime
                    # Try to parse the date string
                    if isinstance(task.due_date, str):
                        due_date_value = datetime.strptime(task.due_date, '%Y-%m-%d').date()
                    else:
                        due_date_value = task.due_date
                except (ValueError, TypeError) as e:
                    print(f"Date parsing error: {e}")
                    due_date_value = None
            
            # Resolve IDs - handle both assignee and assignee_github_username
            assignee_username = getattr(task, 'assignee_github_username', None) or getattr(task, 'assignee', None)
            assignee_user_id = get_user_id_by_github_username(conn, assignee_username)
            repository_id = get_repository_id(conn, owner, repo)
            
            insert_stmt = insert(project_tasks).values(
                title=task.title,
                description=task.description,
                assignee_github_username=assignee_username,  # Use mapped field name
                assignee_user_id=assignee_user_id,  # Resolved user ID
                priority=priority_enum.value,  # Convert enum to string
                status=status_enum.value,  # Convert enum to string
                due_date=str(due_date_value) if due_date_value else None,  # Store as string
                repository_id=repository_id,  # Resolved repository ID
                repo_owner=owner,
                repo_name=repo,                is_completed=False,  # Default to False for new tasks
                created_by=current_user.github_username,
                created_by_user_id=current_user.id  # Use user ID if available
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
                "assignee": created_task.assignee_github_username,  # Use correct field
                "priority": created_task.priority,  # Should be string already
                "status": created_task.status,  # Should be string already  
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
            if task_update.priority == "LOW":
                priority_enum = TaskPriority.LOW
            elif task_update.priority == "HIGH":
                priority_enum = TaskPriority.HIGH
                
            status_enum = TaskStatus.TODO
            if task_update.status == "IN_PROGRESS":
                status_enum = TaskStatus.IN_PROGRESS
            elif task_update.status == "DONE":
                status_enum = TaskStatus.DONE            # Resolve assignee user ID if assignee changed
            assignee_user_id = get_user_id_by_github_username(conn, task_update.assignee)
            
            # Update task
            update_stmt = update(project_tasks).where(
                project_tasks.c.id == task_id
            ).values(
                title=task_update.title,
                description=task_update.description,
                assignee_github_username=task_update.assignee,  # Use correct field name
                assignee_user_id=assignee_user_id,  # Resolved user ID
                priority=priority_enum.value,  # Convert enum to string
                status=status_enum.value,  # Convert enum to string
                due_date=task_update.due_date,
                is_completed=(status_enum == TaskStatus.DONE)  # Set is_completed based on status
            )
            
            conn.execute(update_stmt)
            conn.commit()
              # Get updated task
            updated_task = conn.execute(check_query).fetchone()
            
            return {
                "id": updated_task.id,
                "title": updated_task.title,
                "description": updated_task.description,
                "assignee": updated_task.assignee_github_username,  # Use correct field
                "priority": updated_task.priority,  # Already string
                "status": updated_task.status,  # Already string
                "due_date": updated_task.due_date,
                "repo_owner": updated_task.repo_owner,
                "repo_name": updated_task.repo_name,
                "created_at": updated_task.created_at,
                "updated_at": updated_task.updated_at
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

# ================= TASK MANAGEMENT APIs - Direct Database Access =================

# Pydantic models bổ sung cho các API mới
class TaskStats(BaseModel):
    total_tasks: int
    todo_tasks: int
    in_progress_tasks: int
    done_tasks: int
    high_priority_tasks: int
    medium_priority_tasks: int
    low_priority_tasks: int
    overdue_tasks: int

class BulkTaskCreate(BaseModel):
    tasks: List[TaskCreate]

class BulkTaskUpdate(BaseModel):
    task_ids: List[int]
    updates: TaskUpdate

# Helper functions for resolving IDs
def get_user_id_by_github_username(conn, github_username: str) -> Optional[int]:
    """Get user ID from github username"""
    try:
        from db.models.users import users
        query = select(users.c.id).where(users.c.github_username == github_username)
        result = conn.execute(query).fetchone()
        print(f"Debug: Looking for user '{github_username}', found result: {result}")
        return result[0] if result else None
    except Exception as e:
        print(f"Error getting user ID: {e}")
        return None

def get_repository_id(conn, owner: str, repo_name: str) -> Optional[int]:
    """Get repository ID from owner and name"""
    try:
        from db.models.repositories import repositories
        query = select(repositories.c.id).where(
            and_(
                repositories.c.owner == owner,
                repositories.c.name == repo_name
            )
        )
        result = conn.execute(query).fetchone()
        print(f"Debug: Looking for repository '{owner}/{repo_name}', found result: {result}")
        return result[0] if result else None
    except Exception as e:
        print(f"Error getting repository ID: {e}")
        return None

@router.get("/tasks", response_model=List[TaskResponse])
async def get_all_tasks(
    limit: Optional[int] = Query(100, description="Limit number of results"),
    offset: Optional[int] = Query(0, description="Offset for pagination"),
    status: Optional[str] = Query(None, description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    assignee: Optional[str] = Query(None, description="Filter by assignee"),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lấy tất cả tasks từ database với filtering và pagination"""
    try:
        with engine.connect() as conn:
            # Base query
            query = select(project_tasks)
            
            # Apply filters
            conditions = []
            if status:
                if status == "TODO":
                    conditions.append(project_tasks.c.status == "TODO")
                elif status == "IN_PROGRESS":
                    conditions.append(project_tasks.c.status == "IN_PROGRESS")
                elif status == "DONE":
                    conditions.append(project_tasks.c.status == "DONE")
            
            if priority:
                if priority == "LOW":
                    conditions.append(project_tasks.c.priority == "LOW")
                elif priority == "MEDIUM":
                    conditions.append(project_tasks.c.priority == "MEDIUM")
                elif priority == "HIGH":
                    conditions.append(project_tasks.c.priority == "HIGH")
            
            if assignee:
                conditions.append(project_tasks.c.assignee_github_username == assignee)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            # Apply pagination and ordering
            query = query.order_by(project_tasks.c.created_at.desc()).limit(limit).offset(offset)
            
            result = conn.execute(query)
            tasks = []
            
            for row in result:
                task_dict = {
                    "id": row.id,
                    "title": row.title,
                    "description": row.description,
                    "assignee": row.assignee_github_username,  # Use correct field name
                    "priority": row.priority if row.priority else "MEDIUM",  # Already string
                    "status": row.status if row.status else "TODO",  # Already string
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
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/tasks/stats", response_model=TaskStats)
async def get_task_statistics(
    repo_owner: Optional[str] = Query(None, description="Filter by repository owner"),
    repo_name: Optional[str] = Query(None, description="Filter by repository name"),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lấy thống kê tasks từ database"""
    try:
        with engine.connect() as conn:
            # Base query with optional repo filtering
            base_conditions = []
            if repo_owner:
                base_conditions.append(project_tasks.c.repo_owner == repo_owner)
            if repo_name:
                base_conditions.append(project_tasks.c.repo_name == repo_name)
            
            # Total tasks
            total_query = select([func.count(project_tasks.c.id)])
            if base_conditions:
                total_query = total_query.where(and_(*base_conditions))
            total_tasks = conn.execute(total_query).scalar() or 0
            
            # Tasks by status
            todo_query = select([func.count(project_tasks.c.id)]).where(
                project_tasks.c.status == TaskStatus.TODO
            )
            if base_conditions:
                todo_query = todo_query.where(and_(*base_conditions))
            todo_tasks = conn.execute(todo_query).scalar() or 0
            
            in_progress_query = select([func.count(project_tasks.c.id)]).where(
                project_tasks.c.status == TaskStatus.IN_PROGRESS
            )
            if base_conditions:
                in_progress_query = in_progress_query.where(and_(*base_conditions))
            in_progress_tasks = conn.execute(in_progress_query).scalar() or 0
            
            done_query = select([func.count(project_tasks.c.id)]).where(
                project_tasks.c.status == TaskStatus.DONE
            )
            if base_conditions:
                done_query = done_query.where(and_(*base_conditions))
            done_tasks = conn.execute(done_query).scalar() or 0
            
            # Tasks by priority
            high_priority_query = select([func.count(project_tasks.c.id)]).where(
                project_tasks.c.priority == TaskPriority.HIGH
            )
            if base_conditions:
                high_priority_query = high_priority_query.where(and_(*base_conditions))
            high_priority_tasks = conn.execute(high_priority_query).scalar() or 0
            
            medium_priority_query = select([func.count(project_tasks.c.id)]).where(
                project_tasks.c.priority == TaskPriority.MEDIUM
            )
            if base_conditions:
                medium_priority_query = medium_priority_query.where(and_(*base_conditions))
            medium_priority_tasks = conn.execute(medium_priority_query).scalar() or 0
            
            low_priority_query = select([func.count(project_tasks.c.id)]).where(
                project_tasks.c.priority == TaskPriority.LOW
            )
            if base_conditions:
                low_priority_query = low_priority_query.where(and_(*base_conditions))
            low_priority_tasks = conn.execute(low_priority_query).scalar() or 0
            
            # Overdue tasks (tasks with due_date < today and status != done)
            from datetime import date
            today = date.today()
            overdue_conditions = [
                project_tasks.c.due_date < today,
                project_tasks.c.status != TaskStatus.DONE
            ]
            if base_conditions:
                overdue_conditions.extend(base_conditions)
            
            overdue_query = select([func.count(project_tasks.c.id)]).where(
                and_(*overdue_conditions)
            )
            overdue_tasks = conn.execute(overdue_query).scalar() or 0
            
            return {
                "total_tasks": total_tasks,
                "todo_tasks": todo_tasks,
                "in_progress_tasks": in_progress_tasks,
                "done_tasks": done_tasks,
                "high_priority_tasks": high_priority_tasks,
                "medium_priority_tasks": medium_priority_tasks,
                "low_priority_tasks": low_priority_tasks,
                "overdue_tasks": overdue_tasks
            }
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/tasks/by-assignee/{assignee}", response_model=List[TaskResponse])
async def get_tasks_by_assignee(
    assignee: str,
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: Optional[int] = Query(50, description="Limit number of results"),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lấy tất cả tasks được giao cho một người cụ thể"""
    try:
        with engine.connect() as conn:
            conditions = [project_tasks.c.assignee == assignee]
            
            if status:
                if status == "todo":
                    conditions.append(project_tasks.c.status == TaskStatus.TODO)
                elif status == "in_progress":
                    conditions.append(project_tasks.c.status == TaskStatus.IN_PROGRESS)
                elif status == "done":
                    conditions.append(project_tasks.c.status == TaskStatus.DONE)
            
            query = select(project_tasks).where(
                and_(*conditions)
            ).order_by(project_tasks.c.created_at.desc()).limit(limit)
            
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
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.post("/tasks/bulk", response_model=List[TaskResponse])
async def create_bulk_tasks(
    bulk_data: BulkTaskCreate,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Tạo nhiều tasks cùng lúc"""
    try:
        created_tasks = []
        
        with engine.connect() as conn:
            for task in bulk_data.tasks:
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
                    repo_owner=task.repo_owner,
                    repo_name=task.repo_name,
                    created_by=current_user["username"]
                )
                
                result = conn.execute(insert_stmt)
                task_id = result.inserted_primary_key[0]
                
                # Get the created task
                query = select(project_tasks).where(project_tasks.c.id == task_id)
                created_task = conn.execute(query).fetchone()
                
                created_tasks.append({
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
                })
            
            conn.commit()
            
        return created_tasks
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.put("/tasks/bulk-update")
async def bulk_update_tasks(
    bulk_update: BulkTaskUpdate,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cập nhật nhiều tasks cùng lúc"""
    try:
        with engine.connect() as conn:
            # Validate priority and status
            updates = {}
            
            if bulk_update.updates.title:
                updates["title"] = bulk_update.updates.title
            if bulk_update.updates.description:
                updates["description"] = bulk_update.updates.description
            if bulk_update.updates.assignee:
                updates["assignee"] = bulk_update.updates.assignee
            
            if bulk_update.updates.priority:
                priority_enum = TaskPriority.MEDIUM
                if bulk_update.updates.priority == "low":
                    priority_enum = TaskPriority.LOW
                elif bulk_update.updates.priority == "high":
                    priority_enum = TaskPriority.HIGH
                updates["priority"] = priority_enum
                
            if bulk_update.updates.status:
                status_enum = TaskStatus.TODO
                if bulk_update.updates.status == "in_progress":
                    status_enum = TaskStatus.IN_PROGRESS
                elif bulk_update.updates.status == "done":
                    status_enum = TaskStatus.DONE
                updates["status"] = status_enum
            
            if bulk_update.updates.due_date:
                updates["due_date"] = bulk_update.updates.due_date
            
            # Update tasks
            update_stmt = update(project_tasks).where(
                project_tasks.c.id.in_(bulk_update.task_ids)
            ).values(**updates)
            
            result = conn.execute(update_stmt)
            conn.commit()
            
            return {
                "message": f"Successfully updated {result.rowcount} tasks",
                "updated_count": result.rowcount,
                "task_ids": bulk_update.task_ids
            }
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/tasks/search")
async def search_tasks(
    q: str = Query(..., description="Search query"),
    limit: Optional[int] = Query(50, description="Limit number of results"),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Tìm kiếm tasks theo từ khóa"""
    try:
        with engine.connect() as conn:
            # Search in title, description, and assignee
            search_term = f"%{q}%"
            query = select(project_tasks).where(
                or_(
                    project_tasks.c.title.ilike(search_term),
                    project_tasks.c.description.ilike(search_term),
                    project_tasks.c.assignee.ilike(search_term)
                )
            ).order_by(project_tasks.c.created_at.desc()).limit(limit)
            
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
            
            return {
                "query": q,
                "results": tasks,
                "count": len(tasks)
            }
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")