import asyncio
import csv
from decimal import Decimal
from pathlib import Path
import sys

import structlog # For logging
from sqlalchemy import delete # To clear the table
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Adjust path to import from src
# This assumes the script is run from the project root directory (e.g., `python claims_processor/scripts/data/load_rvu_data.py`)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from claims_processor.src.core.config.settings import get_settings
from claims_processor.src.core.database.models.rvu_data_db import RVUDataModel # RVU DB Model
# If using structlog, basic setup:
# from claims_processor.src.monitoring.logging.logging_config import setup_logging
# setup_logging() # Call if you want structured logging output like the main app

logger = structlog.get_logger(__name__)

async def load_rvu_data():
    settings = get_settings()
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    AsyncSessionFactory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Path to the CSV data file, relative to this script's parent's parent (project root)
    # Script is at: claims_processor/scripts/data/load_rvu_data.py
    # Project root is Path(__file__).resolve().parents[2]
    # Data file is at: data/rvu_data.csv
    csv_file_path = Path(__file__).resolve().parents[2] / "data" / "rvu_data.csv"

    if not csv_file_path.is_file():
        logger.error("RVU data CSV file not found.", path=str(csv_file_path))
        return

    logger.info("Starting RVU data loading process...", csv_file=str(csv_file_path))

    async with AsyncSessionFactory() as session:
        async with session.begin():
            # Clear existing data from rvu_data table
            logger.info(f"Clearing existing data from {RVUDataModel.__tablename__} table...")
            await session.execute(delete(RVUDataModel))
            logger.info(f"Existing data cleared from {RVUDataModel.__tablename__}.")

            rvu_records_to_add = []
            try:
                with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        try:
                            record = RVUDataModel(
                                procedure_code=row['procedure_code'],
                                rvu_value=Decimal(row['rvu_value']),
                                description=row.get('description') # Handle if description is optional
                            )
                            rvu_records_to_add.append(record)
                        except KeyError as e:
                            logger.error(f"Missing expected column in CSV row: {e}. Row: {row}")
                            continue # Skip malformed row
                        except Exception as e:
                            logger.error(f"Error processing row: {row}. Error: {e}", exc_info=True)
                            continue # Skip problematic row


                if rvu_records_to_add:
                    session.add_all(rvu_records_to_add)
                    await session.commit() # Commit within the begin block for the main data load
                    logger.info(f"Successfully loaded {len(rvu_records_to_add)} RVU records into the database.")
                else:
                    logger.warn("No valid RVU records found in CSV to load.")

            except FileNotFoundError:
                logger.error(f"RVU data file not found at {csv_file_path}")
            except Exception as e:
                await session.rollback() # Rollback on any error during file processing or db insertion
                logger.error(f"An error occurred during RVU data loading: {e}", exc_info=True)


if __name__ == "__main__":
    # Basic logging setup for script execution (if not using the app's full setup_logging)
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logger.info("Running RVU data loading script...")
    asyncio.run(load_rvu_data())
    logger.info("RVU data loading script finished.")
