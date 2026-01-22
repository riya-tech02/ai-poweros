"""Dashboard routes"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pathlib import Path

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve basic dashboard"""
    html_path = Path(__file__).parent.parent.parent / "web" / "templates" / "dashboard.html"
    if html_path.exists():
        with open(html_path, 'r') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard not found</h1>")

@router.get("/advanced", response_class=HTMLResponse)
async def advanced_dashboard():
    """Serve advanced analytics dashboard"""
    html_path = Path(__file__).parent.parent.parent / "web" / "templates" / "advanced_dashboard.html"
    if html_path.exists():
        with open(html_path, 'r') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Advanced dashboard not found</h1>")
