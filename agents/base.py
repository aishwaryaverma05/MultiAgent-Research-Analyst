
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """
    Common typed interface for all research agents.

    Concrete agents will receive dependencies through their constructors.
    """

    name: str

    @abstractmethod
    async def run(self, payload: InputT) -> OutputT:
        """Execute an agent and return validated output."""
        raise NotImplementedError
