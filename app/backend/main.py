
"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.backend.routes.research import router as research_router
from config import get_settings
from database.connection import Base, engine
from database.models import ResearchTask  # noqa: F401

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database tables when the application starts."""

    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Multi-Agent Research Analyst API",
    lifespan=lifespan,
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
    """Check database-independent API health."""
    return {
        "status": "healthy",
    }
