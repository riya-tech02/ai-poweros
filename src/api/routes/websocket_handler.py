"""
Real-time WebSocket handler for live predictions
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import asyncio
import json
from datetime import datetime

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process real-time prediction request
            response = {
                "type": "prediction",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "activity": "Real-time prediction",
                    "confidence": 0.95
                }
            }
            
            await websocket.send_json(response)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
