from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, Optional

class SyncEventBase(BaseModel):
    repo_key: str
    event_type: str
    data: Optional[Dict[str, Any]] = None

class SyncEventCreate(SyncEventBase):
    pass

class SyncEventRead(SyncEventBase):
    id: int
    timestamp: datetime

    class Config:
        orm_mode = True
