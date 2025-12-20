import sys
import os
from pathlib import Path
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

current_dir = Path(__file__).parent
app_dir = current_dir.parent
project_root = app_dir.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(app_dir))

from dotenv import load_dotenv

env_local = project_root / '.env.local'
env_default = project_root / '.env'

if env_local.exists():
    load_dotenv(env_local)
elif env_default.exists():
    load_dotenv(env_default)

from app.database.session import Base
from app.database.models.route_logs_model import RouteLog
from app.database.models.users_model import User

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def get_db_url():
    migration_url = os.getenv("MIGRATION_DATABASE_URL")

    return migration_url


db_url = get_db_url()
config.set_main_option("sqlalchemy.url", db_url)

target_metadata = Base.metadata


def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()