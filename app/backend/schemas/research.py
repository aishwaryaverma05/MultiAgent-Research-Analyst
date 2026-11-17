
"""Pydantic schemas for research API requests and responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Request schema for starting a research task."""

    query: str = Field(
        min_length=3,
        max_length=2000,
        description="Research question or topic",
    )


class ResearchResponse(BaseModel):
    """Response schema for a research task."""

    task_id: str
    status: str
    message: str


class ResearchStatusResponse(BaseModel):
    """Response schema for checking research task status."""

    task_id: str
    query: str
    status: str
    result: dict[str, Any] | None = None
    created_at: datetime


class ResearchApprovalRequest(BaseModel):
    """Request schema for approving or rejecting a research report."""

    approved: bool
    feedback: str = Field(
        default="",
        max_length=5000,
        description="Optional feedback for report revision or rejection.",
    )
