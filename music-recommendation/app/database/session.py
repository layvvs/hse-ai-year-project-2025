from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.core.config import parse_database_config
from typing import AsyncGenerator

Base = declarative_base()

db_config = parse_database_config()

engine = create_async_engine(
    db_config.database_url,
    echo=False,
)

session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency для FastAPI
    session: AsyncSession = Depends(get_db)
    """
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def dispose_engine():
    await engine.dispose()
