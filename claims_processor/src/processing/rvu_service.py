import csv # For parsing CSV
from pathlib import Path # For robust path handling
from sqlalchemy.ext.asyncio import AsyncSession
from decimal import Decimal
import structlog
from typing import Optional, Dict # Added Dict

from ..api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
from ..core.cache.cache_manager import CacheManager
# from ..core.config.settings import get_settings # No longer needed directly here for file path
from ..core.database.models.rvu_data_db import RVUDataModel # Import DB Model
from sqlalchemy import select # Import select
from ..core.monitoring.metrics import CACHE_OPERATIONS_TOTAL # Import the metric

logger = structlog.get_logger(__name__)

MEDICARE_CONVERSION_FACTOR = Decimal("38.87") # This could also be moved to settings

class RVUService:
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.conversion_factor = MEDICARE_CONVERSION_FACTOR
        # self.rvu_data_map is no longer loaded at init. Data is fetched from DB on cache miss.
        logger.info("RVUService initialized with CacheManager. RVU data will be fetched from DB.", cache_manager_id=id(cache_manager))

    async def calculate_rvu_for_claim(self, claim: ProcessableClaim, db_session: AsyncSession):
        logger.info("Calculating RVUs for claim (with cache and DB data)", claim_id=claim.claim_id, db_claim_id=claim.id)

        if not claim.line_items:
            logger.warn("Claim has no line items for RVU calculation", claim_id=claim.claim_id)
            return

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
