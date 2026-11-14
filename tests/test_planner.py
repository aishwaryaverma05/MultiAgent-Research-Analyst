
import pytest

from agents.contracts import (
    ResearchPlan,
    ResearchQuestion,
    ResearchRequest,
    SearchTask,
)
from agents.planner import PlannerAgent


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.received_prompt = None

    async def ainvoke(self, prompt):
        self.received_prompt = prompt
        return self.response


class FakeChatModel:
    def __init__(self, response):
        self.structured_model = FakeStructuredModel(response)

    def with_structured_output(self, schema):
        assert schema is ResearchPlan
        return self.structured_model


def create_valid_plan():
    return ResearchPlan(
        objective="Understand AI agents in healthcare.",
        research_questions=[
            ResearchQuestion(
                question="How are AI agents used in healthcare?"
            )
        ],
        search_tasks=[
            SearchTask(
                question="Investigate clinical applications.",
                query="AI agents clinical healthcare applications",
            ),
            SearchTask(
                question="Investigate healthcare risks.",
                query="AI agents healthcare risks and limitations",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_planner_returns_research_plan():
    fake_model = FakeChatModel(create_valid_plan())
    planner = PlannerAgent(model=fake_model)

    result = await planner.run(
        ResearchRequest(
            query="How are AI agents transforming healthcare?"
        )
    )

    assert isinstance(result, ResearchPlan)
    assert len(result.search_tasks) == 2
    assert result.objective


@pytest.mark.asyncio
async def test_planner_accepts_dictionary_output():
    fake_model = FakeChatModel(create_valid_plan().model_dump())
    planner = PlannerAgent(model=fake_model)

    result = await planner.run(
        ResearchRequest(
            query="How are AI agents transforming healthcare?"
        )
    )

    assert isinstance(result, ResearchPlan)
    assert len(result.search_tasks) == 2


@pytest.mark.asyncio
async def test_planner_rejects_empty_search_tasks():
    invalid_plan = create_valid_plan().model_copy(
        update={"search_tasks": []}
    )

    fake_model = FakeChatModel(invalid_plan)
    planner = PlannerAgent(model=fake_model)

    with pytest.raises(ValueError, match="no executable search tasks"):
        await planner.run(
            ResearchRequest(
                query="How are AI agents transforming healthcare?"
            )
        )
