
from datetime import UTC, datetime

from agents.contracts import (
    CriticRequest,
    EvidenceDocument,
    FactCheckRequest,
    ResearchReport,
    ResearchRequest,
    WriteRequest,
)
from agents.critic import CriticAgent
from agents.fact_checker import FactCheckerAgent
from agents.supervisor import SupervisorAgent
from agents.writer import WriterAgent
from graph.state import ResearchState


def initialize_research(state: ResearchState) -> dict:
    """Validate the request and initialize workflow metadata."""

    query = state.get("query", "").strip()

    if len(query) < 8:
        raise ValueError(
            "Research query must contain at least 8 characters."
        )

    return {
        "status": "running",
        "current_stage": "research_initialized",
        "created_at": (
            state.get("created_at")
            or datetime.now(UTC).isoformat()
        ),
        "revision_count": state.get("revision_count", 0),
        "max_revisions": state.get("max_revisions", 3),
        "errors": [],
    }


def make_supervisor_node(supervisor: SupervisorAgent):
    """Create a graph node that executes the Supervisor Agent."""

    async def run_supervisor(state: ResearchState) -> dict:
        request = ResearchRequest(
            query=state["query"],
            user_id=state.get("user_id"),
        )

        result = await supervisor.run(request)

        return {
            "plan": result.plan.model_dump(),
            "sources": [
                source.model_dump() for source in result.sources
            ],
            "evidence": [
                item.model_dump() for item in result.evidence
            ],
            "supervisor_status": result.status,
            "errors": list(result.errors),
            "current_stage": "supervisor_completed",
        }

    return run_supervisor


def make_writer_node(writer: WriterAgent):
    """Create a graph node that generates the research report."""

    async def run_writer(state: ResearchState) -> dict:
        evidence = [
            EvidenceDocument.model_validate(item)
            for item in state.get("evidence", [])
        ]

        if not evidence:
            raise ValueError(
                "Cannot generate a report without research evidence."
            )

        previous_report = state.get("report", {})

        request = WriteRequest(
            query=state["query"],
            evidence=evidence,
            previous_report=previous_report.get("markdown") or None,
            revision_feedback=state.get("critic_feedback", []),
        )

        report = await writer.run(request)

        return {
            "report": report.model_dump(),
            "current_stage": "writer_completed",
        }

    return run_writer


def make_critic_node(critic: CriticAgent):
    """Create a graph node that evaluates the research report."""

    async def run_critic(state: ResearchState) -> dict:
        report = state.get("report")

        if not report:
            raise ValueError("Cannot evaluate an empty research report.")

        evidence = [
            EvidenceDocument.model_validate(item)
            for item in state.get("evidence", [])
        ]

        request = CriticRequest(
            report=ResearchReport.model_validate(report),
            evidence=evidence,
        )

        evaluation = await critic.run(request)

        feedback = (
            evaluation.revision_instructions
            + evaluation.unsupported_claims
            + evaluation.citation_issues
            + evaluation.missing_information
        )

        return {
            "critic_evaluation": evaluation.model_dump(),
            "critic_score": evaluation.overall_score,
            "critic_feedback": feedback,
            "current_stage": "critic_completed",
        }

    return run_critic


def prepare_revision(state: ResearchState) -> dict:
    """Prepare feedback and increment the revision counter."""

    return {
        "revision_count": state.get("revision_count", 0) + 1,
        "current_stage": "revision_prepared",
    }


def make_fact_checker_node(fact_checker: FactCheckerAgent):
    """Create a graph node that verifies report claims."""

    async def run_fact_checker(state: ResearchState) -> dict:
        report_data = state.get("report")

        if not report_data:
            raise ValueError(
                "Cannot fact-check an empty research report."
            )

        evidence = [
            EvidenceDocument.model_validate(item)
            for item in state.get("evidence", [])
        ]

        

        request = FactCheckRequest(
            report=ResearchReport.model_validate(report_data),
            evidence=evidence,
        )

        result = await fact_checker.run(request)

        has_issues = bool(
            result.unsupported_claims
            or result.contradictory_claims
            or result.unknown_citations
        )

        return {
            "fact_check": result.model_dump(),
            "status": "partial" if has_issues else "completed",
            "current_stage": "fact_check_completed",
        }

    return run_fact_checker
