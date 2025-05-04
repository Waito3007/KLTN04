from pydantic import BaseModel
from datetime import datetime


class CommitCreate(BaseModel):
    commit_id: str
    message: str
    author_name: str
    author_email: str
    committed_date: datetime
    repository_id: int


class CommitOut(CommitCreate):
    id: int

    class Config:
        from_attributes = True  # Dành cho Pydantic V2 thay cho orm_mode
