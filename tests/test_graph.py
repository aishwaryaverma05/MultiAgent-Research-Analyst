
import pytest

from agents.contracts import (
    CriticEvaluation,
    EvidenceDocument,
    EvidenceItem,
    FactCheckResult,
    ResearchPlan,
    ResearchQuestion,
    ResearchReport,
    SourceRecord,
    SupervisorResult,
)
from graph.nodes import initialize_research
from graph.workflow import build_workflow


def make_evidence():
    return EvidenceDocument(
        source_id="source-001",
        title="Renewable Energy Report",
        url="https://example.com/report",
        text="Solar energy capacity increased significantly.",
        evidence_items=[
            EvidenceItem(
                claim="Solar energy capacity increased.",
                supporting_quote=(
                    "Solar energy capacity increased significantly."
                ),
            )
        ],
        extraction_status="success",
    )


def make_report():
    return ResearchReport(
        title="Renewable Energy Research",
        executive_summary="Solar energy capacity is growing.",
        sections=[
            {
                "heading": "Solar Energy",
                "content": "Solar capacity has increased.",
            }
        ],
        limitations=["Limited source coverage."],
        references=["https://example.com/report"],
        markdown="# Renewable Energy Research\n\nSolar capacity is growing.",
    )


def make_supervisor_result():
    source = SourceRecord(
        source_id="source-001",
        title="Renewable Energy Report",
        url="https://example.com/report",
        snippet="Renewable energy research information.",
    )

    return SupervisorResult(
        plan=ResearchPlan(
            objective="Research renewable energy development",
            research_questions=[
                ResearchQuestion(
                    question="How is solar energy capacity growing?"
                )
            ],
        ),
        sources=[source],
        evidence=[make_evidence()],
        errors=[],
        status="success",
        total_search_tasks=1,
        successful_search_tasks=1,
        failed_search_tasks=0,
        total_sources=1,
        attempted_readers=1,
        failed_readers=0,
    )


def make_critic_evaluation(score):
    return CriticEvaluation(
        overall_score=score,
        dimension_scores={
            "factual_accuracy": score,
            "evidence_quality": score,
            "citation_quality": score,
            "completeness": score,
            "clarity": score,
        },
        unsupported_claims=[],
        citation_issues=[],
        missing_information=[],
        revision_instructions=(
            ["Improve the report."] if score < 7 else []
        ),
    )


class FakeSupervisor:
    async def run(self, payload):
        return make_supervisor_result()


class FakeWriter:
    def __init__(self):
        self.calls = 0

    async def run(self, payload):
        self.calls += 1
        return make_report()


class FakeCritic:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    async def run(self, payload):
        score = self.scores[self.calls]
        self.calls += 1
        return make_critic_evaluation(score)


class FakeFactChecker:
    def __init__(self):
        self.calls = 0

    async def run(self, payload):
        self.calls += 1
        return FactCheckResult(
            verified_claims=[
                "Solar energy capacity increased."
            ],
            unsupported_claims=[],
            contradictory_claims=[],
            unknown_citations=[],
        )


def make_workflow(critic_scores):
    writer = FakeWriter()
    critic = FakeCritic(critic_scores)
    fact_checker = FakeFactChecker()

    workflow = build_workflow(
        supervisor=FakeSupervisor(),
        writer=writer,
        critic=critic,
        fact_checker=fact_checker,
    )

    return workflow, writer, critic, fact_checker


@pytest.mark.asyncio
async def test_complete_workflow_completes_successfully():
    workflow, writer, critic, fact_checker = make_workflow([8.5])

    result = await workflow.ainvoke(
        {
            "research_id": "test-001",
            "query": "Evaluate modern RAG architectures",
        }
    )

    assert result["status"] == "completed"
    assert result["current_stage"] == "fact_check_completed"
    assert result["research_id"] == "test-001"
    assert result["critic_score"] == 8.5
    assert result["fact_check"]["verified_claims"]
    assert writer.calls == 1
    assert critic.calls == 1
    assert fact_checker.calls == 1


@pytest.mark.asyncio
async def test_workflow_revises_report_when_critic_score_is_low():
    workflow, writer, critic, fact_checker = make_workflow(
        [6.0, 8.5]
    )

    result = await workflow.ainvoke(
        {
            "research_id": "test-002",
            "query": "Evaluate modern RAG architectures",
        }
    )

    assert result["status"] == "completed"
    assert result["revision_count"] == 1
    assert result["critic_score"] == 8.5
    assert writer.calls == 2
    assert critic.calls == 2
    assert fact_checker.calls == 1


@pytest.mark.asyncio
async def test_workflow_stops_revision_at_maximum():
    workflow, writer, critic, fact_checker = make_workflow(
        [4.0, 4.0, 4.0, 4.0]
    )

    result = await workflow.ainvoke(
        {
            "research_id": "test-003",
            "query": "Evaluate modern RAG architectures",
            "max_revisions": 3,
        }
    )

    assert result["status"] == "completed"
    assert result["revision_count"] == 3
    assert writer.calls == 4
    assert critic.calls == 4
    assert fact_checker.calls == 1


def test_short_query_is_rejected():
    with pytest.raises(ValueError):
        initialize_research(
            {
                "query": "short",
            }
        )
