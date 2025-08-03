from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, List, Any
import json
import asyncio
import logging
from datetime import datetime

sync_events_router = APIRouter()
logger = logging.getLogger(__name__)

class SyncEventManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.sync_events: Dict[str, List[Dict]] = {}  # repo_key -> events (in-memory for now)
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast_event(self, event: Dict[str, Any]):
        """Broadcast sync event to all connected clients"""
        if not self.active_connections:
            return
            
        message = json.dumps(event)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def add_sync_event(self, repo_key: str, event_type: str, data: Dict = None):
        """Add sync event and broadcast to clients"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "repo_key": repo_key,
            "event_type": event_type,
            "data": data or {}
        }
        
        # Store event in memory
        if repo_key not in self.sync_events:
            self.sync_events[repo_key] = []
        
        self.sync_events[repo_key].append(event)
        
        # Keep only last 50 events per repo
        if len(self.sync_events[repo_key]) > 50:
            self.sync_events[repo_key] = self.sync_events[repo_key][-50:]
        
        # Broadcast to connected clients
        await self.broadcast_event(event)
        
        logger.info(f"Sync event added: {event_type} for {repo_key}")
        
        # Log important events to existing tables (update last_synced, sync_status, etc)
        await self._log_to_existing_tables(repo_key, event_type, data)
    
    async def _log_to_existing_tables(self, repo_key: str, event_type: str, data: Dict = None):
        """Log sync events to existing database tables"""
        try:
            # Only log for major events to avoid cluttering
            if event_type in ['sync_start', 'sync_complete', 'sync_error']:
                # Here you can update repository table, add to a sync_log field, etc.
                # For now, just log it
                logger.info(f"📝 Logged {event_type} for {repo_key} to existing tables")
                
                # Example: Update repository sync status
                # owner, repo_name = repo_key.split('/')
                # Update repository last_synced, sync_status fields in database
                
        except Exception as e:
            logger.error(f"❌ Error logging to existing tables: {e}")
    
    def get_repo_events(self, repo_key: str) -> List[Dict]:
        """Get sync events for a specific repository"""
        return self.sync_events.get(repo_key, [])
    
    def get_all_events(self) -> Dict[str, List[Dict]]:
        """Get all sync events"""
        return self.sync_events

# Global instance
sync_event_manager = SyncEventManager()

# Remove the decorator since we'll add it directly in main.py
# @sync_events_router.websocket("/ws")
async def websocket_sync_events(websocket: WebSocket):
    """WebSocket endpoint for real-time sync events"""
    logger.info("🔌 WebSocket connection attempt received")
    
    try:
        await websocket.accept()
        logger.info("✅ WebSocket connection accepted")
        
        # Add to manager
        sync_event_manager.active_connections.append(websocket)
        logger.info(f"📊 Total WebSocket connections: {len(sync_event_manager.active_connections)}")
        
        # Send welcome message
        welcome_msg = {
            "type": "connection_established",
            "message": "WebSocket connected successfully",
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send_text(json.dumps(welcome_msg))
        
        while True:
            try:
                # Keep connection alive and listen for client messages
                data = await websocket.receive_text()
                logger.info(f"📨 Received WebSocket message: {data}")
                
                # Echo back with server info
                response = {
                    "type": "echo",
                    "received": data,
                    "timestamp": datetime.now().isoformat(),
                    "active_connections": len(sync_event_manager.active_connections)
                }
                await websocket.send_text(json.dumps(response))
                
            except Exception as e:
                logger.error(f"❌ Error in WebSocket message loop: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"❌ WebSocket connection error: {e}")
    finally:
        # Clean up connection
        if websocket in sync_event_manager.active_connections:
            sync_event_manager.active_connections.remove(websocket)
        logger.info(f"🧹 WebSocket cleaned up. Remaining connections: {len(sync_event_manager.active_connections)}")

@sync_events_router.get("/repositories/{owner}/{repo}/events")
async def get_repo_sync_events(owner: str, repo: str):
    """Get sync events for a specific repository"""
    repo_key = f"{owner}/{repo}"
    events = sync_event_manager.get_repo_events(repo_key)
    
    return {
        "repo_key": repo_key,
        "events": events,
        "total_events": len(events)
    }

@sync_events_router.get("/sync-events")
async def get_all_sync_events():
    """Get all sync events"""
    all_events = sync_event_manager.get_all_events()
    
    return {
        "events": all_events,
        "total_repos": len(all_events),
        "total_events": sum(len(events) for events in all_events.values())
    }

@sync_events_router.delete("/repositories/{owner}/{repo}/events")
async def clear_repo_sync_events(owner: str, repo: str):
    """Clear sync events for a specific repository"""
    repo_key = f"{owner}/{repo}"
    
    try:
        deleted_count = 0
        if repo_key in sync_event_manager.sync_events:
            deleted_count = len(sync_event_manager.sync_events[repo_key])
            sync_event_manager.sync_events[repo_key] = []
        
        # Broadcast clear event
        await sync_event_manager.broadcast_event({
            "timestamp": datetime.now().isoformat(),
            "repo_key": repo_key,
            "event_type": "events_cleared",
            "data": {"deleted_count": deleted_count}
        })
        
        return {"message": f"Cleared {deleted_count} events for {repo_key}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing events: {str(e)}")

@sync_events_router.get("/sync-events/stats")
async def get_sync_events_stats():
    """Get sync events statistics"""
    try:
        all_events = sync_event_manager.get_all_events()
        
        total_events = sum(len(events) for events in all_events.values())
        repos_with_events = len(all_events)
        
        # Count events by type
        event_types = {}
        for events in all_events.values():
            for event in events:
                event_type = event.get('event_type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1
        
        stats = {
            "total_events": total_events,
            "repos_with_events": repos_with_events,
            "event_types": event_types,
            "active_connections": len(sync_event_manager.active_connections)
        }
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@sync_events_router.delete("/sync-events/cleanup")
async def cleanup_old_sync_events(days_back: int = 7):
    """Clean up old sync events (admin endpoint)"""
    try:
        # For in-memory storage, keep only last 20 events per repo
        deleted_count = 0
        
        for repo_key in sync_event_manager.sync_events:
            events = sync_event_manager.sync_events[repo_key]
            if len(events) > 20:
                deleted_count += len(events) - 20
                sync_event_manager.sync_events[repo_key] = events[-20:]
        
        return {"message": f"Cleaned up {deleted_count} old events (keeping last 20 per repo)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up events: {str(e)}")

# Helper functions for other modules to emit events
async def emit_sync_start(repo_key: str, sync_type: str = "optimized"):
    """Emit sync start event"""
    await sync_event_manager.add_sync_event(
        repo_key, 
        "sync_start", 
        {"sync_type": sync_type}
    )

async def emit_sync_progress(repo_key: str, current: int, total: int, stage: str):
    """Emit sync progress event"""
    await sync_event_manager.add_sync_event(
        repo_key,
        "sync_progress",
        {
            "current": current,
            "total": total,
            "percentage": round((current / total) * 100, 1) if total > 0 else 0,
            "stage": stage
        }
    )

async def emit_sync_complete(repo_key: str, success: bool, stats: Dict = None):
    """Emit sync complete event"""
    await sync_event_manager.add_sync_event(
        repo_key,
        "sync_complete",
        {
            "success": success,
            "stats": stats or {}
        }
    )

async def emit_sync_error(repo_key: str, error: str, stage: str = None):
    """Emit sync error event"""
    await sync_event_manager.add_sync_event(
        repo_key,
        "sync_error",
        {
            "error": error,
            "stage": stage
        }
    )
