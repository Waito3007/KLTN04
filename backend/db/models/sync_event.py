from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.sql import func
from db.database import Base
import json
from datetime import datetime

class SyncEvent(Base):
    __tablename__ = "sync_events"
    
    id = Column(Integer, primary_key=True, index=True)
    repo_key = Column(String(255), nullable=False, index=True)  # owner/repo format
    event_type = Column(String(50), nullable=False)  # sync_start, sync_progress, sync_complete, sync_error
    event_data = Column(Text, nullable=True)  # JSON data
    timestamp = Column(DateTime, nullable=False, default=func.now())
    created_at = Column(DateTime, nullable=False, default=func.now())
    
    def __repr__(self):
        return f"<SyncEvent(id={self.id}, repo_key='{self.repo_key}', event_type='{self.event_type}')>"
    
    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "repo_key": self.repo_key,
            "event_type": self.event_type,
            "data": json.loads(self.event_data) if self.event_data else {},
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, repo_key: str, event_type: str, data: dict = None, timestamp: datetime = None):
        """Create SyncEvent from dictionary"""
        return cls(
            repo_key=repo_key,
            event_type=event_type,
            event_data=json.dumps(data) if data else None,
            timestamp=timestamp or datetime.now()
        )
