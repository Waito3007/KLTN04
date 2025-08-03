from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from db.models.sync_event import SyncEvent
from db.database import get_db
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class SyncEventService:
    def __init__(self):
        pass
    
    async def create_sync_event(
        self, 
        db: Session, 
        repo_key: str, 
        event_type: str, 
        data: Dict = None,
        timestamp: datetime = None
    ) -> SyncEvent:
        """Create and save a sync event to database"""
        try:
            sync_event = SyncEvent.from_dict(
                repo_key=repo_key,
                event_type=event_type,
                data=data,
                timestamp=timestamp
            )
            
            db.add(sync_event)
            db.commit()
            db.refresh(sync_event)
            
            logger.info(f"✅ Sync event saved to DB: {event_type} for {repo_key}")
            return sync_event
            
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error saving sync event to DB: {e}")
            raise
    
    async def get_repo_events(
        self, 
        db: Session, 
        repo_key: str, 
        limit: int = 50,
        hours_back: int = 24
    ) -> List[SyncEvent]:
        """Get recent sync events for a repository"""
        try:
            # Get events from last 24 hours by default
            since_time = datetime.now() - timedelta(hours=hours_back)
            
            events = db.query(SyncEvent).filter(
                and_(
                    SyncEvent.repo_key == repo_key,
                    SyncEvent.timestamp >= since_time
                )
            ).order_by(desc(SyncEvent.timestamp)).limit(limit).all()
            
            return events
            
        except Exception as e:
            logger.error(f"❌ Error getting repo events from DB: {e}")
            return []
    
    async def get_all_recent_events(
        self, 
        db: Session, 
        limit_per_repo: int = 20,
        hours_back: int = 24
    ) -> Dict[str, List[Dict]]:
        """Get recent events for all repositories"""
        try:
            since_time = datetime.now() - timedelta(hours=hours_back)
            
            # Get all recent events
            events = db.query(SyncEvent).filter(
                SyncEvent.timestamp >= since_time
            ).order_by(desc(SyncEvent.timestamp)).all()
            
            # Group by repo_key
            grouped_events = {}
            repo_counts = {}
            
            for event in events:
                repo_key = event.repo_key
                
                if repo_key not in grouped_events:
                    grouped_events[repo_key] = []
                    repo_counts[repo_key] = 0
                
                # Limit events per repo
                if repo_counts[repo_key] < limit_per_repo:
                    grouped_events[repo_key].append(event.to_dict())
                    repo_counts[repo_key] += 1
            
            return grouped_events
            
        except Exception as e:
            logger.error(f"❌ Error getting all events from DB: {e}")
            return {}
    
    async def clear_repo_events(self, db: Session, repo_key: str) -> int:
        """Clear all events for a repository"""
        try:
            deleted_count = db.query(SyncEvent).filter(
                SyncEvent.repo_key == repo_key
            ).delete()
            
            db.commit()
            logger.info(f"🗑️ Cleared {deleted_count} events for {repo_key}")
            return deleted_count
            
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error clearing repo events: {e}")
            raise
    
    async def cleanup_old_events(self, db: Session, days_back: int = 7) -> int:
        """Clean up events older than specified days"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            deleted_count = db.query(SyncEvent).filter(
                SyncEvent.timestamp < cutoff_time
            ).delete()
            
            db.commit()
            logger.info(f"🧹 Cleaned up {deleted_count} old sync events (older than {days_back} days)")
            return deleted_count
            
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error cleaning up old events: {e}")
            raise
    
    async def get_event_stats(self, db: Session) -> Dict:
        """Get statistics about sync events"""
        try:
            total_events = db.query(SyncEvent).count()
            
            # Count by event type
            event_type_counts = {}
            event_types = db.query(SyncEvent.event_type).distinct().all()
            
            for (event_type,) in event_types:
                count = db.query(SyncEvent).filter(
                    SyncEvent.event_type == event_type
                ).count()
                event_type_counts[event_type] = count
            
            # Count unique repos
            unique_repos = db.query(SyncEvent.repo_key).distinct().count()
            
            # Recent activity (last 24 hours)
            recent_time = datetime.now() - timedelta(hours=24)
            recent_events = db.query(SyncEvent).filter(
                SyncEvent.timestamp >= recent_time
            ).count()
            
            return {
                "total_events": total_events,
                "unique_repositories": unique_repos,
                "recent_events_24h": recent_events,
                "event_type_counts": event_type_counts
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting event stats: {e}")
            return {}

# Global service instance
sync_event_service = SyncEventService()
