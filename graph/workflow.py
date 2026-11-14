
from langgraph.graph import END, START, StateGraph

from agents.critic import CriticAgent
from agents.fact_checker import FactCheckerAgent
from agents.supervisor import SupervisorAgent
from agents.writer import WriterAgent
from graph.nodes import (
    initialize_research,
    make_critic_node,
    make_fact_checker_node,
    make_supervisor_node,
    make_writer_node,
    prepare_revision,
)
from graph.routing import route_after_critic
from graph.state import ResearchState


def route_critic_result(state: ResearchState) -> str:
    """Choose revision or fact-checking based on critic score."""

    return route_after_critic(
        score=state.get("critic_score", 0.0),
        revision_count=state.get("revision_count", 0),
        max_revisions=state.get("max_revisions", 3),
    )


def build_workflow(
    supervisor: SupervisorAgent | None = None,
    writer: WriterAgent | None = None,
    critic: CriticAgent | None = None,
    fact_checker: FactCheckerAgent | None = None,
):
    """Build and compile the complete multi-agent research workflow."""

    supervisor = supervisor or SupervisorAgent()
    writer = writer or WriterAgent()
    critic = critic or CriticAgent()
    fact_checker = fact_checker or FactCheckerAgent()

    builder = StateGraph(ResearchState)

    # Register workflow nodes
    builder.add_node("initialize_research", initialize_research)
    builder.add_node("supervisor", make_supervisor_node(supervisor))
    builder.add_node("writer", make_writer_node(writer))
    builder.add_node("critic", make_critic_node(critic))
    builder.add_node("prepare_revision", prepare_revision)
    builder.add_node(
        "fact_checker",
        make_fact_checker_node(fact_checker),
    )

    # Main workflow
    builder.add_edge(START, "initialize_research")
    builder.add_edge("initialize_research", "supervisor")
    builder.add_edge("supervisor", "writer")
    builder.add_edge("writer", "critic")

    # Critic decides whether revision is required
    builder.add_conditional_edges(
        "critic",
        route_critic_result,
        {
            "revise": "prepare_revision",
            "human_review": "fact_checker",
        },
    )

    # Revision loop
    builder.add_edge("prepare_revision", "writer")

    # Final verification
    builder.add_edge("fact_checker", END)

    return builder.compile()
