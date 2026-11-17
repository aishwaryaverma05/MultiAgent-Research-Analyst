
"""Smoke test for the PostgreSQL LangGraph checkpointer."""

import asyncio

from database.checkpointer import get_checkpointer


async def main() -> None:
    """Verify PostgreSQL checkpointer initialization."""

    async with get_checkpointer() as checkpointer:
        print("PostgreSQL checkpointer initialized successfully")
        print("Checkpointer type:", type(checkpointer).__name__)


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=asyncio.SelectorEventLoop,
    )
