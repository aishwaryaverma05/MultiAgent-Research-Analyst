
"""PostgreSQL checkpoint manager for LangGraph workflows."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.engine import make_url

from config import get_settings


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """Create and manage a PostgreSQL-backed LangGraph checkpointer."""

    settings = get_settings()

    database_url = make_url(settings.database_url)
    connection_string = (
        database_url
        .set(drivername="postgresql")
        .render_as_string(hide_password=False)
    )

    async with AsyncConnectionPool(
        conninfo=connection_string,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
        },
        open=False,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)

        await checkpointer.setup()

        yield checkpointer
