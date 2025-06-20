import os
import sys
from sqlalchemy.ext.asyncio import create_async_engine # For async engine - keep for offline or if needed elsewhere
from logging.config import fileConfig
import sqlalchemy # Add this import

# Add project src to Python path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'claims_processor', 'src'))) # Relying on alembic.ini prepend_sys_path

from claims_processor.src.core.config.settings import get_settings
from claims_processor.src.core.database.db_session import Base # Import Base from your db_session
# Explicitly import all model classes that inherit from Base for Alembic's metadata
from claims_processor.src.core.database.models.claims_db import ClaimModel, ClaimLineItemModel
from claims_processor.src.core.database.models.audit_log_db import AuditLogModel
from claims_processor.src.core.database.models.claims_production_db import ClaimsProductionModel
from claims_processor.src.core.database.models.rvu_data_db import RVUDataModel # Add this


from sqlalchemy import pool # Keep this, it might be used by engine_from_config if that path is taken

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    settings = get_settings()
    url = settings.DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True, # Add this for better enum/boolean/etc. detection
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    settings = get_settings()
    sync_db_url = settings.DATABASE_URL.replace("postgresql+asyncpg", "postgresql")

    # Create a synchronous engine for Alembic's online mode
    connectable = sqlalchemy.create_engine(sync_db_url)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # If you have schemas, include them here:
            # version_table_schema=target_metadata.schema,
            # include_schemas=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
