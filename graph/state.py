
from typing import Literal, TypedDict


class ResearchState(TypedDict, total=False):
    # Research identification
    research_id: str
    user_id: str | None

    # User request
    query: str

    # Workflow tracking
    status: Literal[
        "pending",
        "running",
        "awaiting_approval",
        "completed",
        "failed",
        "partial",
    ]
    current_stage: str

    # Supervisor outputs
    plan: dict
    sources: list[dict]
    evidence: list[dict]
    supervisor_status: str

    # Writer output
    report: dict

    # Critic output
    critic_evaluation: dict
    critic_score: float
    critic_feedback: list[str]

    # Fact checker output
    fact_check: dict

    # Error tracking
    errors: list[str]

    # Revision management
    revision_count: int
    max_revisions: int

    # Metadata
    created_at: str
