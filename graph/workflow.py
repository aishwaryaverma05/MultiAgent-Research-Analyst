from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.critic import CriticAgent
from agents.fact_checker import FactCheckerAgent
from agents.supervisor import SupervisorAgent
from agents.writer import WriterAgent
from graph.nodes import (
    finalize_rejection,
    human_approval_node,
    initialize_research,
    make_critic_node,
    make_fact_checker_node,
    make_supervisor_node,
    make_writer_node,
    mark_awaiting_approval,
    prepare_revision,
)
from graph.routing import route_after_critic
from graph.state import ResearchState


def route_critic_result(state: ResearchState) -> str:
    """Choose revision or human approval based on critic score."""

    return route_after_critic(
        score=state.get("critic_score", 0.0),
        revision_count=state.get("revision_count", 0),
        max_revisions=state.get("max_revisions", 3),
    )


def route_human_decision(state: ResearchState) -> str:
    """Route the workflow based on the human approval decision."""

    approval = state.get("human_approval", {})
    approved = approval.get("approved", False)

    if approved:
        return "approved"

    if state.get("revision_count", 0) < state.get("max_revisions", 3):
        return "revise"

    return "rejected"


def build_workflow(
    supervisor: SupervisorAgent | None = None,
    writer: WriterAgent | None = None,
    critic: CriticAgent | None = None,
    fact_checker: FactCheckerAgent | None = None,
    checkpointer=None,
):
    """Build and compile the complete human-in-the-loop research workflow."""

    supervisor = supervisor or SupervisorAgent()
    writer = writer or WriterAgent()
    critic = critic or CriticAgent()
    fact_checker = fact_checker or FactCheckerAgent()

    if checkpointer is None:
        checkpointer = InMemorySaver()

    builder = StateGraph(ResearchState)

    # Register workflow nodes
    builder.add_node("initialize_research", initialize_research)
    builder.add_node("supervisor", make_supervisor_node(supervisor))
    builder.add_node("writer", make_writer_node(writer))
    builder.add_node("critic", make_critic_node(critic))
    builder.add_node("prepare_revision", prepare_revision)
    builder.add_node(
        "mark_awaiting_approval",
        mark_awaiting_approval,
    )
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("finalize_rejection", finalize_rejection)
    builder.add_node(
        "fact_checker",
        make_fact_checker_node(fact_checker),
    )

    # Main workflow
    builder.add_edge(START, "initialize_research")
    builder.add_edge("initialize_research", "supervisor")
    builder.add_edge("supervisor", "writer")
    builder.add_edge("writer", "critic")

    # Critic decides whether revision or human approval is required
    builder.add_conditional_edges(
        "critic",
        route_critic_result,
        {
            "revise": "prepare_revision",
            "human_review": "mark_awaiting_approval",
        },
    )

    # Human approval flow
    builder.add_edge("mark_awaiting_approval", "human_approval")

    builder.add_conditional_edges(
        "human_approval",
        route_human_decision,
        {
            "approved": "fact_checker",
            "revise": "prepare_revision",
            "rejected": "finalize_rejection",
        },
    )

    # Revision loop
    builder.add_edge("prepare_revision", "writer")

    # Final outcomes
    builder.add_edge("fact_checker", END)
    builder.add_edge("finalize_rejection", END)

    return builder.compile(checkpointer=checkpointer)
