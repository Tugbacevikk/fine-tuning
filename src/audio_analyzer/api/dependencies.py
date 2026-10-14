import logging

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from audio_analyzer.adapters.repository.postgres_repository import PostgresRepository
from audio_analyzer.adapters.repository.session import (
    AsyncSessionLocal,
    SessionLocal,
    create_async_db_engine,
    engine,
    get_async_db_url,
    get_uow,
    get_uow_with_engine,
    init_engine,
)
from audio_analyzer.adapters.repository.unit_of_work import SqlAlchemyUnitOfWork
from audio_analyzer.domain.interfaces import ITranscriptRepository

logger = logging.getLogger(__name__)

__all__ = [
    "get_async_db_url",
    "create_async_db_engine",
    "init_engine",
    "engine",
    "AsyncSessionLocal",
    "SessionLocal",
    "get_db",
    "get_repository",
    "get_uow",
    "get_uow_with_engine",
    "SqlAlchemyUnitOfWork",
]


async def get_db():
    """FastAPI Asenkron Veritabanı Oturumu (AsyncSession) Bağımlılığı."""
    async with AsyncSessionLocal() as session:
        yield session


async def get_repository(db: AsyncSession = Depends(get_db)) -> ITranscriptRepository:
    """FastAPI Bağımlılık Enjeksiyonu ile ITranscriptRepository örneği sağlar."""
    return PostgresRepository(session=db)
