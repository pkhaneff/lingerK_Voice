"""Database session configuration"""
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from urllib.parse import quote_plus
from loguru import logger as custom_logger
from app.core.config import DB_DATABASE, DB_HOST, DB_PASSWORD, DB_PORT, DB_USER_NAME

def get_connection_url() -> str:
    """Constructs the database connection URL"""
    return f"postgresql+asyncpg://{quote_plus(str(DB_USER_NAME))}:{quote_plus(str(DB_PASSWORD))}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"

# Engine Configuration
engine = create_async_engine(
    get_connection_url(),
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
    pool_use_lifo=True,
    echo=False
)

# Session Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

custom_logger.info("Database connection pool initialized")
