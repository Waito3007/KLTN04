from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, List, Any
import json
import asyncio
import logging
from datetime import datetime

from db.database import get_db
from services.sync_event_service import sync_event_service, SyncEventService
from schemas.sync_event import SyncEventCreate, SyncEventRead

sync_events_router = APIRouter()
logger = logging.getLogger(__name__)

class SyncEventManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast_event(self, event: Dict[str, Any]):
        if not self.active_connections:
            return
        
        # Ensure event is JSON serializable
        message = json.dumps(event, default=str)
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                disconnected.append(connection)
        
        for connection in disconnected:
            self.disconnect(connection)

sync_event_manager = SyncEventManager()

@sync_events_router.websocket("/ws")
async def websocket_sync_events(websocket: WebSocket):
    await sync_event_manager.connect(websocket)
    try:
        welcome_msg = {
            "type": "connection_established",
            "message": "WebSocket connected successfully",
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send_text(json.dumps(welcome_msg))
        
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")
            # Echo back for testing
            response = {
                "type": "echo",
                "received": data,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        sync_event_manager.disconnect(websocket)

@sync_events_router.get("/repositories/{owner}/{repo}/events", response_model=List[SyncEventRead])
async def get_repo_sync_events(owner: str, repo: str, db: Session = Depends(get_db)):
    repo_key = f"{owner}/{repo}"
    events = await sync_event_service.get_repo_events(db, repo_key)
    return events

@sync_events_router.get("/sync-events", response_model=Dict[str, List[SyncEventRead]])
async def get_all_sync_events(db: Session = Depends(get_db)):
    events = await sync_event_service.get_all_recent_events(db)
    return events

@sync_events_router.delete("/repositories/{owner}/{repo}/events")
async def clear_repo_sync_events(owner: str, repo: str, db: Session = Depends(get_db)):
    repo_key = f"{owner}/{repo}"
    deleted_count = await sync_event_service.clear_repo_events(db, repo_key)
    
    await sync_event_manager.broadcast_event({
        "repo_key": repo_key,
        "event_type": "events_cleared",
        "data": {"deleted_count": deleted_count},
        "timestamp": datetime.now().isoformat()
    })
    return {"message": f"Cleared {deleted_count} events for {repo_key}"}

@sync_events_router.get("/sync-events/stats")
async def get_sync_events_stats(db: Session = Depends(get_db)):
    stats = await sync_event_service.get_event_stats(db)
    stats["active_connections"] = len(sync_event_manager.active_connections)
    return stats

@sync_events_router.delete("/sync-events/cleanup")
async def cleanup_old_sync_events(days_back: int = 7, db: Session = Depends(get_db)):
    deleted_count = await sync_event_service.cleanup_old_events(db, days_back)
    return {"message": f"Cleaned up {deleted_count} old events"}

# Helper functions for other modules to emit events
async def emit_sync_event(db: Session, repo_key: str, event_type: str, data: Dict = None):
    # 1. Save event to database
    event_data = SyncEventCreate(repo_key=repo_key, event_type=event_type, data=data)
    db_event = await sync_event_service.create_sync_event(db, event_data)
    
    # 2. Broadcast to WebSocket clients
    await sync_event_manager.broadcast_event(db_event.to_dict())

async def emit_sync_start(db: Session, repo_key: str, sync_type: str = "optimized"):
    await emit_sync_event(db, repo_key, "sync_start", {"sync_type": sync_type})

async def emit_sync_progress(db: Session, repo_key: str, current: int, total: int, stage: str):
    data = {
        "current": current,
        "total": total,
        "percentage": round((current / total) * 100, 1) if total > 0 else 0,
        "stage": stage
    }
    await emit_sync_event(db, repo_key, "sync_progress", data)

async def emit_sync_complete(db: Session, repo_key: str, success: bool, stats: Dict = None):
    data = {"success": success, "stats": stats or {}}
    await emit_sync_event(db, repo_key, "sync_complete", data)

async def emit_sync_error(db: Session, repo_key: str, error: str, stage: str = None):
    data = {"error": error, "stage": stage}
    await emit_sync_event(db, repo_key, "sync_error", data)
