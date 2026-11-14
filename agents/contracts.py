## This file defines the input and output structures for our agents.

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class ResearchRequest(StrictModel):
    query: str = Field(min_length=8, max_length=2000)
    user_id: str | None = None


class ResearchQuestion(StrictModel):
    question: str = Field(min_length=5)
    rationale: str = ""

class SearchTask(StrictModel):
    question: str = Field(min_length=5)
    query: str = Field(min_length=5)

class ResearchPlan(StrictModel):
    objective: str = Field(min_length=10)

    research_questions: list[ResearchQuestion] = Field(
        min_length=1,
        max_length=12,
    )

    # Executable tasks that Search Workers will consume.
    # Default keeps older ResearchPlan construction compatible.
    search_tasks: list[SearchTask] = Field(
        default_factory=list,
        max_length=12,
    )

    subtopics: list[str] = Field(default_factory=list)
    search_strategy: list[str] = Field(default_factory=list)
    expected_findings: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)




class SourceRecord(StrictModel):
    source_id: str
    title: str
    url: str
    snippet: str = ""
    relevance_score: float | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class SearchTaskFailure(StrictModel):
    """Details of an individual search task failure."""

    task_index: int
    query: str
    error: str


class SearchBatchResult(StrictModel):
    """Aggregated output from parallel search workers."""

    sources: list[SourceRecord] = Field(default_factory=list)
    failures: list[SearchTaskFailure] = Field(default_factory=list)
    total_tasks: int = 0
    successful_tasks: int = 0

class ReadRequest(StrictModel):
    source: SourceRecord



class EvidenceItem(StrictModel):
    """One factual claim supported by text from a source."""

    claim: str = Field(min_length=10)
    supporting_quote: str = Field(min_length=10)


class EvidenceExtraction(StrictModel):
    """Structured output requested from the Reader LLM."""

    evidence_items: list[EvidenceItem] = Field(
        default_factory=list,
        max_length=30,
    )


class EvidenceDocument(StrictModel):
    """Extracted evidence with its original source attribution."""

    source_id: str
    url: str
    title: str
    text: str
    evidence_items: list[EvidenceItem] = Field(
        default_factory=list,
    )
    extraction_status: Literal[
        "pending",
        "success",
        "partial",
        "failed",
    ] = "pending"


class SupervisorResult(StrictModel):
    """Complete result of the supervised research execution."""

    plan: ResearchPlan

    sources: list[SourceRecord] = Field(default_factory=list)

    evidence: list[EvidenceDocument] = Field(default_factory=list)

    errors: list[str] = Field(default_factory=list)

    status: Literal["success", "partial", "failed"]

    total_search_tasks: int = 0
    successful_search_tasks: int = 0
    failed_search_tasks: int = 0

    total_sources: int = 0
    attempted_readers: int = 0
    failed_readers: int = 0

class WriteRequest(StrictModel):
    query: str
    evidence: list[EvidenceDocument] = Field(default_factory=list)
    previous_report: str | None = None
    revision_feedback: list[str] = Field(default_factory=list)


class ResearchReport(StrictModel):
    title: str
    executive_summary: str
    sections: list[dict[str, str]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    markdown: str = ""


class CriticRequest(StrictModel):
    report: ResearchReport
    evidence: list[EvidenceDocument] = Field(default_factory=list)


class CriticEvaluation(StrictModel):
    overall_score: float = Field(ge=0, le=10)
    dimension_scores: dict[str, float]

    unsupported_claims: list[str] = Field(default_factory=list)
    citation_issues: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)


class FactCheckRequest(StrictModel):
    report: ResearchReport
    evidence: list[EvidenceDocument] = Field(default_factory=list)


class FactCheckResult(StrictModel):
    verified_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictory_claims: list[str] = Field(default_factory=list)
    unknown_citations: list[str] = Field(default_factory=list)
