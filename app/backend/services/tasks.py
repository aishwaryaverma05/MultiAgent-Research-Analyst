
"""Celery background tasks for research execution."""

import asyncio
import json
import logging

from celery import Task
from langgraph.types import Command

from app.backend.celery_app import celery_app
from database.checkpointer import get_checkpointer
from database.connection import SessionLocal
from database.models import ResearchTask
from graph.workflow import build_workflow

logger = logging.getLogger(__name__)


async def _execute_research(task_id: str, query: str) -> dict:
    """Execute the LangGraph research workflow asynchronously."""

    async with get_checkpointer() as checkpointer:
        workflow = build_workflow(checkpointer=checkpointer)

        initial_state = {
            "research_id": task_id,
            "query": query,
            "status": "pending",
        }

        return await workflow.ainvoke(
            initial_state,
            config={
                "configurable": {
                    "thread_id": task_id,
                }
            },
        )


async def _resume_research(
    task_id: str,
    approved: bool,
    feedback: str,
) -> dict:
    """Resume a paused LangGraph workflow after human review."""

    async with get_checkpointer() as checkpointer:
        workflow = build_workflow(checkpointer=checkpointer)

        return await workflow.ainvoke(
            Command(
                resume={
                    "approved": approved,
                    "feedback": feedback,
                }
            ),
            config={
                "configurable": {
                    "thread_id": task_id,
                }
            },
        )


def _save_workflow_result(
    research_task: ResearchTask,
    result: dict,
) -> dict:
    """Save the workflow status and report to PostgreSQL."""

    final_status = result.get("status", "running")
    report = result.get("report")

    if final_status not in {
        "completed",
        "partial",
        "rejected",
        "awaiting_approval",
    }:
        final_status = "running"

    research_task.status = final_status

    if report is not None:
        research_task.result = json.dumps(
            report,
            ensure_ascii=False,
            default=str,
        )

    return {
        "task_id": research_task.task_id,
        "status": final_status,
    }


@celery_app.task(
    bind=True,
    name="app.backend.services.tasks.run_research_task",
)
def run_research_task(self: Task, task_id: str, query: str) -> dict:
    """Run a research workflow as a Celery background task."""

    db = SessionLocal()

    try:
        research_task = db.get(ResearchTask, task_id)

        if research_task is None:
            raise ValueError(f"Research task not found: {task_id}")

        research_task.status = "running"
        db.commit()

        result = asyncio.run(
            _execute_research(task_id, query),
            loop_factory=asyncio.SelectorEventLoop,
        )

        response = _save_workflow_result(research_task, result)
        db.commit()

        logger.info(
            "Research task %s finished with status %s",
            task_id,
            response["status"],
        )

        return response

    except Exception:
        db.rollback()

        research_task = db.get(ResearchTask, task_id)

        if research_task is not None:
            research_task.status = "failed"
            db.commit()

        logger.exception("Research task %s failed", task_id)
        raise

    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="app.backend.services.tasks.resume_research_task",
)
def resume_research_task(
    self: Task,
    task_id: str,
    approved: bool,
    feedback: str = "",
) -> dict:
    """Resume a research workflow after human approval or rejection."""

    db = SessionLocal()

    try:
        research_task = db.get(ResearchTask, task_id)

        if research_task is None:
            raise ValueError(f"Research task not found: {task_id}")

        research_task.status = "running"
        db.commit()

        result = asyncio.run(
            _resume_research(task_id, approved, feedback),
            loop_factory=asyncio.SelectorEventLoop,
        )

        response = _save_workflow_result(research_task, result)
        db.commit()

        logger.info(
            "Research task %s resumed and finished with status %s",
            task_id,
            response["status"],
        )

        return response

    except Exception:
        db.rollback()

        research_task = db.get(ResearchTask, task_id)

        if research_task is not None:
            research_task.status = "failed"
            db.commit()

        logger.exception(
            "Failed to resume research task %s",
            task_id,
        )
        raise

    finally:
        db.close()
