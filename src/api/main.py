"""FastAPI application - Complete with Dashboard"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import advanced_prediction, dashboard, prediction
from src.core.config import settings
from src.core.logging import logger, setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI-PowerOS", version="1.0.0")
    yield
    logger.info("Shutting down AI-PowerOS")


app = FastAPI(
    title="AI-PowerOS API",
    version="1.0.0",
    description="Complete AI Personal Operating System",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])

app.include_router(
    prediction.router, prefix="/api/v1/predict", tags=["basic-prediction"]
)

app.include_router(
    advanced_prediction.router, prefix="/api/v1/advanced", tags=["advanced-ml"]
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "features": [
            "routine-prediction",
            "task-scheduling",
            "memory-system",
            "graph-integration",
            "web-dashboard",
        ],
    }


@app.get("/")
async def root():
    return {
        "message": "AI-PowerOS - Complete Operating System",
        "version": "1.0.0",
        "dashboard": "/dashboard",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "dashboard": "/dashboard",
            "basic_predict": "/api/v1/predict/routine",
            "advanced_predict": "/api/v1/advanced/routine/advanced",
            "schedule": "/api/v1/advanced/schedule/intelligent",
            "habits": "/api/v1/advanced/habits/record",
        },
    }
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def desktop():
    return """
    <html>
      <head>
        <title>AI-PowerOS Desktop</title>
        <style>
          body {
            background: #0f172a;
            color: white;
            font-family: system-ui;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
          }
          .card {
            background: #020617;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 0 30px rgba(0,255,255,0.2);
            text-align: center;
          }
          a {
            color: cyan;
            text-decoration: none;
            display: block;
            margin-top: 10px;
          }
        </style>
      </head>
      <body>
        <div class="card">
          <h1>🧠 AI-PowerOS</h1>
          <p>System Status: Online</p>
          <a href="/docs">API Control Panel</a>
          <a href="/dashboard">Dashboard (coming soon)</a>
        </div>
      </body>
    </html>
    """

