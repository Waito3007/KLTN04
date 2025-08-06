from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from db.models.sync_event import SyncEvent
from schemas.sync_event import SyncEventCreate
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SyncEventService:
    async def create_sync_event(self, db: Session, event: SyncEventCreate) -> SyncEvent:
        try:
            db_event = SyncEvent(
                repo_key=event.repo_key,
                event_type=event.event_type,
                data=event.data,
                timestamp=datetime.now()
            )
            db.add(db_event)
            db.commit()
            db.refresh(db_event)
            logger.info(f"✅ Sync event saved to DB: {event.event_type} for {event.repo_key}")
            return db_event
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error saving sync event to DB: {e}")
            raise

    async def get_repo_events(
        self, db: Session, repo_key: str, limit: int = 50, hours_back: int = 24
    ) -> List[SyncEvent]:
        try:
            since_time = datetime.now() - timedelta(hours=hours_back)
            events = (
                db.query(SyncEvent)
                .filter(SyncEvent.repo_key == repo_key, SyncEvent.timestamp >= since_time)
                .order_by(desc(SyncEvent.timestamp))
                .limit(limit)
                .all()
            )
            return events
        except Exception as e:
            logger.error(f"❌ Error getting repo events from DB: {e}")
            return []

    async def get_all_recent_events(
        self, db: Session, limit_per_repo: int = 20, hours_back: int = 48
    ) -> Dict[str, List[Dict]]:
        try:
            since_time = datetime.now() - timedelta(hours=hours_back)
            events = (
                db.query(SyncEvent)
                .filter(SyncEvent.timestamp >= since_time)
                .order_by(desc(SyncEvent.timestamp))
                .all()
            )
            
            grouped_events = {}
            for event in events:
                if event.repo_key not in grouped_events:
                    grouped_events[event.repo_key] = []
                if len(grouped_events[event.repo_key]) < limit_per_repo:
                    grouped_events[event.repo_key].append(event.to_dict())
            
            return grouped_events
        except Exception as e:
            logger.error(f"❌ Error getting all events from DB: {e}")
            return {}

    async def clear_repo_events(self, db: Session, repo_key: str) -> int:
        try:
            deleted_count = db.query(SyncEvent).filter(SyncEvent.repo_key == repo_key).delete()
            db.commit()
            logger.info(f"🗑️ Cleared {deleted_count} events for {repo_key}")
            return deleted_count
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error clearing repo events: {e}")
            raise

    async def cleanup_old_events(self, db: Session, days_back: int = 7) -> int:
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            deleted_count = db.query(SyncEvent).filter(SyncEvent.timestamp < cutoff_time).delete()
            db.commit()
            logger.info(f"🧹 Cleaned up {deleted_count} old sync events")
            return deleted_count
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error cleaning up old events: {e}")
            raise

    async def get_event_stats(self, db: Session) -> Dict:
        try:
            total_events = db.query(SyncEvent).count()
            unique_repos = db.query(SyncEvent.repo_key).distinct().count()
            
            recent_time = datetime.now() - timedelta(hours=24)
            recent_events = db.query(SyncEvent).filter(SyncEvent.timestamp >= recent_time).count()
            
            return {
                "total_events": total_events,
                "unique_repositories": unique_repos,
                "recent_events_24h": recent_events,
            }
        except Exception as e:
            logger.error(f"❌ Error getting event stats: {e}")
            return {}

sync_event_service = SyncEventService()
