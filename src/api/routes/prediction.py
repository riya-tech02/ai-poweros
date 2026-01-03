"""Prediction endpoints with ML models"""
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Import ML models
try:
    from src.memory.memory_system import memory_system
    from src.models.routine_predictor import routine_model
    from src.models.task_scheduler import task_scheduler

    MODEL_LOADED = True
except ImportError as e:
    print(f"Warning: Could not import ML models: {e}")
    routine_model = None  # type: ignore
    task_scheduler = None  # type: ignore
    memory_system = None  # type: ignore
    MODEL_LOADED = False


class PredictionRequest(BaseModel):
    user_id: str
    context: Dict
    top_k: int = 5


class PredictionResponse(BaseModel):
    predictions: List[Dict]
    latency_ms: float
    backend: str


class TaskScheduleRequest(BaseModel):
    user_id: str
    tasks: List[Dict]


class TaskScheduleResponse(BaseModel):
    scheduled_tasks: List[Dict]
    completion_rate: float


class MemoryRequest(BaseModel):
    content: str
    context: Dict


@router.post("/routine", response_model=PredictionResponse)
async def predict_routine(request: PredictionRequest) -> PredictionResponse:
    """Predict user's next routine activities"""
    start_time = time.time()

    if not MODEL_LOADED or routine_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get predictions
    recent = request.context.get("recent_activities", [])
    predictions = routine_model.predict(activities=recent, top_k=request.top_k)

    latency = (time.time() - start_time) * 1000

    return PredictionResponse(
        predictions=predictions, latency_ms=round(latency, 2), backend="pytorch"
    )


@router.post("/schedule", response_model=TaskScheduleResponse)
async def schedule_tasks(request: TaskScheduleRequest) -> TaskScheduleResponse:
    """Schedule tasks intelligently"""
    if not MODEL_LOADED or task_scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not loaded")

    # Schedule tasks
    scheduled = task_scheduler.schedule(request.tasks)
    completion_rate = task_scheduler.get_completion_rate(request.tasks)

    return TaskScheduleResponse(
        scheduled_tasks=scheduled, completion_rate=completion_rate
    )


@router.get("/habits/{user_id}")
async def get_user_habits(user_id: str) -> Dict:
    """Get user's habit patterns"""
    return {
        "user_id": user_id,
        "habits": [
            {"name": "Morning Exercise", "frequency": 5, "strength": 0.85},
            {"name": "Coffee Break", "frequency": 7, "strength": 0.95},
            {"name": "Evening Reading", "frequency": 4, "strength": 0.70},
        ],
        "total_habits": 3,
    }


@router.post("/memory/store")
async def store_memory(request: MemoryRequest) -> Dict:
    """Store a memory"""
    if not MODEL_LOADED or memory_system is None:
        raise HTTPException(status_code=503, detail="Memory system not loaded")

    memory_id = memory_system.add_memory(
        content=request.content, context=request.context
    )

    return {"memory_id": memory_id, "status": "stored"}


@router.get("/memory/query")
async def query_memory(query: str, k: int = 5) -> Dict:
    """Query memories"""
    if not MODEL_LOADED or memory_system is None:
        raise HTTPException(status_code=503, detail="Memory system not loaded")

    memories = memory_system.retrieve(query, k)

    return {"query": query, "memories": memories, "count": len(memories)}
