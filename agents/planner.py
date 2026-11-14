
from typing import Any

from agents.base import BaseAgent
from agents.contracts import ResearchPlan, ResearchRequest
from llm.gemini import create_chat_model


class PlannerAgent(BaseAgent[ResearchRequest, ResearchPlan]):
    """Convert a research question into an executable research plan."""

    name = "planner"

    def __init__(self, model: Any | None = None) -> None:
        # Optional model injection makes the agent easy to unit test.
        self._model = model
        self._structured_model = None

    def _get_structured_model(self) -> Any:
        """Create the structured LLM only when the agent is executed."""

        if self._structured_model is None:
            model = self._model or create_chat_model()
            self._structured_model = model.with_structured_output(
                ResearchPlan
            )

        return self._structured_model

    async def run(self, payload: ResearchRequest) -> ResearchPlan:
        """Generate and validate a structured research plan."""

        prompt = f"""
You are an expert research planning agent.

Your job is to convert the user's research question into
a clear, actionable research plan.

USER RESEARCH QUESTION:
{payload.query}

INSTRUCTIONS:

1. Define one clear research objective.
2. Break the question into focused research questions.
3. Create 3 to 6 distinct search tasks.
4. Every search task must contain:
   - question: What information should be investigated?
   - query: A concise, web-search-ready query.
5. Search tasks must cover different aspects of the topic.
6. Include useful subtopics and a search strategy.
7. Identify expected findings and missing information.
8. Do not invent research findings, facts, statistics or sources.
9. Do not answer the research question. Only create the plan.
10. Keep the plan relevant to the user's original question.

Return the result using the required structured schema.
"""

        structured_model = self._get_structured_model()

        result = await structured_model.ainvoke(prompt)

        if isinstance(result, ResearchPlan):
            plan = result
        elif isinstance(result, dict):
            plan = ResearchPlan.model_validate(result)
        else:
            raise TypeError(
                "Planner expected a ResearchPlan or dictionary, "
                f"but received {type(result).__name__}."
            )

        if not plan.search_tasks:
            raise ValueError(
                "Planner generated no executable search tasks."
            )

        return plan
