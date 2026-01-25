"""FastAPI application"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
from src.core.config import settings
from src.core.logging import setup_logging, logger
from src.api.routes import prediction, advanced_prediction
from contextlib import asynccontextmanager

setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI-PowerOS", version="1.0.0")
    yield
    logger.info("Shutting down AI-PowerOS")

app = FastAPI(
    title="AI-PowerOS API",
    version="1.0.0",
    description="AI Personal Operating System",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router, prefix="/api/v1/predict", tags=["prediction"])
app.include_router(advanced_prediction.router, prefix="/api/v1/advanced", tags=["advanced"])

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent.parent / "web" / "templates" / "os_desktop.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>AI-PowerOS</h1><p><a href='/docs'>API Docs</a></p>")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "features": ["transformer-predictions", "rl-scheduling", "memory", "graph"]
    }
