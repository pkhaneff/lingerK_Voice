import os
from urllib.parse import quote_plus
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import app.api.model.audio_model 
import app.api.model.video_model
import app.api.model.semantic_document
from app.api.db.base import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from app.api.db.base import Base
target_metadata = Base.metadata

from app.core.config import DB_DATABASE, DB_HOST, DB_PASSWORD, DB_PORT, DB_USER_NAME

def build_pg_url() -> str:
    host = str(DB_HOST or "")
    port = str(DB_PORT or "5432")
    db   = str(DB_DATABASE or "")
    user = str(DB_USER_NAME or "")
    pwd  = str(DB_PASSWORD or "")
    return f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(pwd)}@{host}:{port}/{db}"

database_url = os.getenv("DATABASE_URL") or build_pg_url()
if not database_url:
    raise RuntimeError("Database URL is empty. Check DB_* config or set DATABASE_URL")

config.set_main_option("sqlalchemy.url", database_url)

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
