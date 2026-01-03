"""Dashboard route"""
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard"""
    html_path = (
        Path(__file__).parent.parent.parent / "web" / "templates" / "dashboard.html"
    )

    if html_path.exists():
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>Dashboard not found</h1>")
