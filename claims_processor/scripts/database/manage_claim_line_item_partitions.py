import argparse
import asyncio
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from pathlib import Path
import sys
import structlog

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Adjust path to import from src
sys.path.append(str(Path(__file__).resolve().parents[2]))

from claims_processor.src.core.config.settings import get_settings

logger = structlog.get_logger(__name__)

PARENT_TABLE_NAME = "claim_line_items"
PARTITION_KEY_COLUMN = "service_date" # As confirmed in analysis

def get_partition_name(year: int, month: int) -> str:
    """Generates the partition table name for a given year and month."""
    return f"{PARENT_TABLE_NAME}_y{year}m{month:02d}"

def generate_create_partition_ddl(year: int, month: int) -> str:
    """Generates the DDL to create a new monthly partition."""
    partition_name = get_partition_name(year, month)

    start_date = date(year, month, 1)
    end_date = start_date + relativedelta(months=1)

    ddl = (
        f"CREATE TABLE IF NOT EXISTS {partition_name} "
        f"PARTITION OF {PARENT_TABLE_NAME} "
        f"FOR VALUES FROM ('{start_date.strftime('%Y-%m-%d')}') "
        f"TO ('{end_date.strftime('%Y-%m-%d')}')"
    )
    return ddl

async def check_partition_exists(session: AsyncSession, partition_name: str) -> bool:
    """Checks if a specific partition table already exists."""
    # Check against pg_class (more direct for tables) or information_schema.tables
    # Using information_schema.tables for broader compatibility if needed, though pg_class is fine for PostgreSQL.
    # Ensure schema is checked if not default 'public'. Assuming 'public' for now.
    query = text(
        "SELECT EXISTS ("
        "   SELECT FROM information_schema.tables "
        "   WHERE table_schema = 'public' AND table_name = :partition_name"
        ");"
    )
    result = await session.execute(query, {"partition_name": partition_name})
    return result.scalar_one()

async def create_partition_if_not_exists(year: int, month: int, execute_ddl: bool = False):
    settings = get_settings()
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    AsyncSessionFactory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    partition_name = get_partition_name(year, month)
    logger.info(f"Managing partition: {partition_name} for {year}-{month:02d}")

    async with AsyncSessionFactory() as session:
        async with session.begin(): # Start a transaction for checks/creation
            exists = await check_partition_exists(session, partition_name)
            if exists:
                logger.info(f"Partition {partition_name} already exists.")
                return

            logger.info(f"Partition {partition_name} does not exist. Generating DDL.")
            ddl_statement = generate_create_partition_ddl(year, month)
            logger.info("Generated DDL", ddl=ddl_statement)

            if execute_ddl:
                logger.info(f"Executing DDL to create partition {partition_name}...")
                try:
                    await session.execute(text(ddl_statement))
                    # No explicit commit needed here if session.begin() handles it,
                    # but for DDL, it's often auto-committing or needs explicit commit outside transaction for some DBs.
                    # For SQLAlchemy with asyncpg, DDL within a transaction block should work and commit with the block.
                    logger.info(f"Successfully created partition {partition_name}.")
                except Exception as e:
                    logger.error(f"Failed to create partition {partition_name}.", error=str(e), exc_info=True)
                    # Rollback will happen due to session.begin() context manager on exception
                    raise
            else:
                logger.info("DDL execution skipped (dry_run=True or execute_ddl=False).")

def main():
    parser = argparse.ArgumentParser(description=f"Manage monthly partitions for the {PARENT_TABLE_NAME} table.")
    parser.add_argument("--year", type=int, required=True, help="Year for the partition (e.g., 2024).")
    parser.add_argument("--month", type=int, required=True, help="Month for the partition (1-12).")
    parser.add_argument("--execute", action="store_true", help="Actually execute the DDL to create the partition. If not set, runs in dry-run mode.")
    parser.add_argument("--setup-logging", action="store_true", help="Enable structured logging output.")

    args = parser.parse_args()

    if args.setup_logging:
        structlog.configure(
            processors=[
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.dev.ConsoleRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    if not (1 <= args.month <= 12):
        logger.error("Invalid month. Month must be between 1 and 12.", specified_month=args.month)
        return

    current_year = datetime.now().year
    if not (current_year - 5 <= args.year <= current_year + 5): # Basic sanity check for year
        logger.warn("Year specified is quite far from current year. Ensure it's correct.", specified_year=args.year, current_year=current_year)


    logger.info(f"Script run for Year: {args.year}, Month: {args.month}, Execute: {args.execute}")
    asyncio.run(create_partition_if_not_exists(args.year, args.month, execute_ddl=args.execute))
    logger.info("Script finished.")

if __name__ == "__main__":
    main()
