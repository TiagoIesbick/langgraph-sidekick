import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def setup_memory() -> AsyncSqliteSaver:
    """Initialize Async SQLite memory storage."""
    db_path = "db/memory.db"
    # open async connection
    conn = await aiosqlite.connect(db_path)
    return AsyncSqliteSaver(conn)
