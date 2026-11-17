"""FastAPI application entry point."""

from fastapi import FastAPI

from app.backend.routes.research import router as research_router
from config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Multi-Agent Research Analyst API",
)

app.include_router(research_router)


@app.get("/")
async def root() -> dict[str, str]:
    """Return basic API information."""
    return {
        "message": "Multi-Agent Research Analyst API",
        "status": "running",
    }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Check API health."""
    return {
        "status": "healthy",
    }