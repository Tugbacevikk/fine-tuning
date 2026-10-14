"""
Asenkron Veritabanı Oturumu, Bağlantı Havuzu ve Unit of Work Altyapısı.
API katmanından bağımsız olarak workers, services ve adapters tarafından doğrudan kullanılabilir.
"""

import logging
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from audio_analyzer.adapters.repository.unit_of_work import SqlAlchemyUnitOfWork

load_dotenv()
logger = logging.getLogger(__name__)


def get_async_db_url(url: str) -> str:
    """
    DB URL'sini async dialect sürücüsüne dönüştürür.
    postgresql:// -> postgresql+asyncpg://
    sqlite:/// -> sqlite+aiosqlite:///
    """
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
    elif url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return url


def create_async_db_engine(db_url: str):
    """
    PostgreSQL (asyncpg) / SQLite (aiosqlite) asenkron veritabanı motoru oluşturan fabrika fonksiyonu.
    PostgreSQL için havuzlama (pool_size, max_overflow, pool_pre_ping) parametrelerini aktif eder.
    """
    async_url = get_async_db_url(db_url)
    if "sqlite" in async_url:
        eng = create_async_engine(async_url, echo=False)

        @event.listens_for(eng.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

        return eng
    else:
        is_testing = os.getenv("TESTING", "false").lower() == "true" or "pytest" in sys.modules
        if is_testing:
            return create_async_engine(async_url, echo=False, poolclass=NullPool)

        # PostgreSQL Kurumsal Asenkron Bağlantı Havuzu
        return create_async_engine(
            async_url,
            echo=False,
            pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "30")),
            pool_recycle=3600,
            pool_pre_ping=True,
        )


def init_engine():
    from audio_analyzer.config import get_settings

    db_url = get_settings().database_url
    logger.info("Veritabanı motoru başlatılıyor: %s", get_async_db_url(db_url))
    return create_async_db_engine(db_url)


engine = init_engine()
AsyncSessionLocal = async_sessionmaker(
    bind=engine, autoflush=False, expire_on_commit=False, class_=AsyncSession
)
SessionLocal = AsyncSessionLocal


def get_uow() -> SqlAlchemyUnitOfWork:
    """Unit of Work örneği sağlar."""
    return SqlAlchemyUnitOfWork(session_factory=AsyncSessionLocal)


def get_uow_with_engine(eng=None) -> SqlAlchemyUnitOfWork:
    """Verilen veya varsayılan engine ile Unit of Work örneği sağlar."""
    if eng is None:
        return get_uow()
    session_factory = async_sessionmaker(
        bind=eng, autoflush=False, expire_on_commit=False, class_=AsyncSession
    )
    return SqlAlchemyUnitOfWork(session_factory=session_factory)
