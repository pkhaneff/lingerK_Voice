"""Database session with optimized connection pool"""
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from urllib.parse import quote_plus
from loguru import logger as custom_logger
from app.core.config import DB_DATABASE, DB_HOST, DB_PASSWORD, DB_PORT, DB_USER_NAME


def build_async_url() -> str:
    """Build PostgreSQL async connection URL"""
    host = str(DB_HOST or "127.0.0.1")
    port = str(DB_PORT or "5432")
    db = str(DB_DATABASE or "BAP_Voice_Identification")
    user = str(DB_USER_NAME or "postgres")
    pwd = str(DB_PASSWORD or "")
    return f"postgresql+asyncpg://{quote_plus(user)}:{quote_plus(pwd)}@{host}:{port}/{db}"


DATABASE_URL = build_async_url()

# Create engine with OPTIMIZED pool settings
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    
    # 🔧 Connection Pool Configuration
    pool_size=20,              # Always keep 20 connections alive
    max_overflow=30,           # Can create 30 more when needed
    pool_timeout=30,           # Wait 30s for connection
    pool_recycle=3600,         # Recycle after 1 hour (prevent stale)
    pool_pre_ping=True,        # Test connection before use
    
    # Performance
    pool_use_lifo=True,        # Reuse recent connections (better cache)
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

custom_logger.info(
    "🗄️ Database pool configured: "
    f"pool_size=20, max_overflow=30, total_max=50 connections"
)
