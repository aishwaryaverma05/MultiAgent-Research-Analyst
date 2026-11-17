
"""API routes for research operations."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.backend.schemas.research import (
    ResearchApprovalRequest,
    ResearchRequest,
    ResearchResponse,
    ResearchStatusResponse,
)
from app.backend.services.tasks import (
    resume_research_task,
    run_research_task,
)
from database.connection import get_db
from database.models import ResearchTask

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["Research"])


@router.post(
    "/",
    response_model=ResearchResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_research(
    request: ResearchRequest,
    db: Session = Depends(get_db),  # noqa: B008
) -> ResearchResponse:
    """Save a research request and enqueue background execution."""

    research_task = ResearchTask(
        query=request.query,
        status="queued",
    )

    db.add(research_task)
    db.commit()
    db.refresh(research_task)

    try:
        run_research_task.delay(
            research_task.task_id,
            research_task.query,
        )
    except Exception as exc:
        logger.exception(
            "Failed to enqueue research task %s",
            research_task.task_id,
        )

        research_task.status = "failed"
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research task could not be queued. Please try again.",
        ) from exc

    return ResearchResponse(
        task_id=research_task.task_id,
        status="queued",
        message=f"Research request queued: {request.query}",
    )


@router.get(
    "/{task_id}",
    response_model=ResearchStatusResponse,
)
async def get_research_status(
    task_id: str,
    db: Session = Depends(get_db),  # noqa: B008
) -> ResearchStatusResponse:
    """Return the current status and result of a research task."""

    research_task = db.get(ResearchTask, task_id)

    if research_task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Research task not found.",
        )

    result = None

    if research_task.result:
        try:
            result = json.loads(research_task.result)
        except json.JSONDecodeError:
            logger.exception(
                "Invalid JSON result for research task %s",
                task_id,
            )

    return ResearchStatusResponse(
        task_id=research_task.task_id,
        query=research_task.query,
        status=research_task.status,
        result=result,
        created_at=research_task.created_at,
    )


@router.post(
    "/{task_id}/approval",
    response_model=ResearchResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_research_approval(
    task_id: str,
    request: ResearchApprovalRequest,
    db: Session = Depends(get_db),  # noqa: B008
) -> ResearchResponse:
    """Submit human approval or rejection for a research report."""

    research_task = db.get(ResearchTask, task_id)

    if research_task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Research task not found.",
        )

    if research_task.status != "awaiting_approval":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Research task is not waiting for human approval. "
                f"Current status: {research_task.status}"
            ),
        )

    try:
        resume_research_task.delay(
            task_id,
            request.approved,
            request.feedback,
        )

        research_task.status = "approval_queued"
        db.commit()

    except Exception as exc:
        db.rollback()

        logger.exception(
            "Failed to queue approval for research task %s",
            task_id,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Approval could not be queued. Please try again.",
        ) from exc

    decision = "approved" if request.approved else "rejected"

    return ResearchResponse(
        task_id=task_id,
        status="approval_queued",
        message=f"Research report {decision}. Workflow resume queued.",
    )
