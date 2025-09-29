from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from urllib.parse import quote_plus
from app.core.config import DB_DATABASE, DB_HOST, DB_PASSWORD, DB_PORT, DB_USER_NAME

def build_async_url() -> str:
    host = str(DB_HOST or "127.0.0.1")
    port = str(DB_PORT or "5432")
    db   = str(DB_DATABASE or "BAP_Voice_Identification")
    user = str(DB_USER_NAME or "postgres")
    pwd  = str(DB_PASSWORD or "")
    return f"postgresql+asyncpg://{quote_plus(user)}:{quote_plus(pwd)}@{host}:{port}/{db}"

DATABASE_URL = build_async_url()  
engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
