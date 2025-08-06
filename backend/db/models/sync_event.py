from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import json

Base = declarative_base()

class SyncEvent(Base):
    __tablename__ = "sync_events"

    id = Column(Integer, primary_key=True, index=True)
    repo_key = Column(String, index=True, nullable=False)
    event_type = Column(String, index=True, nullable=False)
    data = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        """Converts the object to a dictionary."""
        return {
            "id": self.id,
            "repo_key": self.repo_key,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }

    @staticmethod
    def from_dict(repo_key: str, event_type: str, data: dict = None, timestamp: datetime = None):
        """Creates an instance from a dictionary."""
        return SyncEvent(
            repo_key=repo_key,
            event_type=event_type,
            data=data or {},
            timestamp=timestamp or datetime.now()
        )