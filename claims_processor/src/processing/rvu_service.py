import csv # For parsing CSV
from pathlib import Path # For robust path handling
from sqlalchemy.ext.asyncio import AsyncSession
from decimal import Decimal
import structlog
from typing import Optional, Dict # Added Dict

from ..api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
from ..core.cache.cache_manager import CacheManager
from ..core.config.settings import get_settings
from ..core.monitoring.metrics import CACHE_OPERATIONS_TOTAL # Import the metric

logger = structlog.get_logger(__name__)

# MOCK_RVU_DATA is now removed, data will be loaded from CSV.
# DEFAULT_RVU will be handled by the loading logic.
MEDICARE_CONVERSION_FACTOR = Decimal("38.87") # This could also be moved to settings

class RVUService:
    def _load_rvu_data_from_csv(self, file_path: Path) -> Dict[str, Decimal]:
        rvu_map: Dict[str, Decimal] = {}
        logger.info(f"Attempting to load RVU data from CSV: {file_path}")
        try:
            # Assuming the path from settings is relative to the project root,
            # and the application is run from the project root.
            if not file_path.is_file():
                logger.error(f"RVU data file not found at {file_path}. Using minimal default RVU.")
                rvu_map["DEFAULT_RVU"] = Decimal("1.00") # Minimal fallback
                return rvu_map

            with open(file_path, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    proc_code = row.get('procedure_code')
                    rvu_val_str = row.get('rvu_value')
                    if proc_code and rvu_val_str:
                        try:
                            rvu_map[proc_code.strip()] = Decimal(rvu_val_str.strip())
                        except Exception as e:
                            logger.warn(f"Could not parse RVU value for {proc_code}: {rvu_val_str}. Error: {e}")
                    else:
                        logger.warn(f"Skipping row due to missing procedure_code or rvu_value: {row}")

            if "DEFAULT_RVU" not in rvu_map:
                logger.warn(f"DEFAULT_RVU not found in {file_path}. Using hardcoded default of 1.00.")
                rvu_map["DEFAULT_RVU"] = Decimal("1.00")

            logger.info(f"Successfully loaded {len(rvu_map)} RVU records from {file_path}")

        except FileNotFoundError: # This specific exception might be redundant if Path.is_file() check is robust
            logger.error(f"RVU data file not found during open: {file_path}. Using minimal default RVU.")
            if "DEFAULT_RVU" not in rvu_map: rvu_map["DEFAULT_RVU"] = Decimal("1.00")
        except Exception as e:
            logger.error(f"Failed to load RVU data from {file_path}: {e}", exc_info=True)
            if "DEFAULT_RVU" not in rvu_map: rvu_map["DEFAULT_RVU"] = Decimal("1.00")
        return rvu_map

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.conversion_factor = MEDICARE_CONVERSION_FACTOR

        settings = get_settings()
        # Path instances created from relative paths are resolved against the current working directory.
        # This requires the application to be run from the project root for "data/rvu_data.csv" to work.
        rvu_file_path = Path(settings.RVU_DATA_FILE_PATH)

        self.rvu_data_map: Dict[str, Decimal] = self._load_rvu_data_from_csv(rvu_file_path)

        logger.info("RVUService initialized. RVU data loaded.", num_rvu_records=len(self.rvu_data_map), cache_manager_id=id(cache_manager))

    async def calculate_rvu_for_claim(self, claim: ProcessableClaim, db_session: AsyncSession):
        logger.info("Calculating RVUs for claim (with cache and CSV data)", claim_id=claim.claim_id, db_claim_id=claim.id)

        if not claim.line_items:
            logger.warn("Claim has no line items for RVU calculation", claim_id=claim.claim_id)
            return

        default_rvu_value = self.rvu_data_map.get("DEFAULT_RVU", Decimal("1.00")) # Ensure fallback for DEFAULT_RVU itself

        for line_item in claim.line_items:
            procedure_code = line_item.procedure_code
            units = line_item.units

            cache_key = f"rvu:{procedure_code}"
            rvu_per_unit_str = await self.cache_manager.get(cache_key)

            rvu_per_unit: Optional[Decimal] = None

            if rvu_per_unit_str is not None:
                CACHE_OPERATIONS_TOTAL.labels(cache_type='rvu_cache', operation_type='hit').inc() # HIT
                try:
                    rvu_per_unit = Decimal(rvu_per_unit_str)
                    logger.debug("RVU cache hit", procedure_code=procedure_code, rvu_value=rvu_per_unit)
                except Exception as e:
                    logger.warn("Failed to parse cached RVU value, falling back to CSV/default.",
                                procedure_code=procedure_code, cached_value=rvu_per_unit_str, error=str(e))
                    rvu_per_unit = None # Treat parse error as miss, force re-fetch from source

            if rvu_per_unit is None: # Cache miss OR cache hit but failed to parse
                CACHE_OPERATIONS_TOTAL.labels(cache_type='rvu_cache', operation_type='miss').inc() # MISS
                logger.debug("RVU cache miss or parse error, using data from CSV/default.", procedure_code=procedure_code)

                rvu_per_unit = self.rvu_data_map.get(procedure_code, default_rvu_value) # Use loaded map
                # Set in cache (CacheManager.set now internally handles set_error metric)
                await self.cache_manager.set(cache_key, str(rvu_per_unit), ttl=3600)

            line_item.rvu_total = rvu_per_unit * Decimal(units)

            logger.debug(f"Line {line_item.line_number}: Proc {procedure_code}, Units {units}, RVU/Unit {rvu_per_unit}, Total RVU {line_item.rvu_total}")

        logger.info("Finished RVU calculation for claim (with cache and CSV data)", claim_id=claim.claim_id, db_claim_id=claim.id)
