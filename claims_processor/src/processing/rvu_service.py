import csv # For parsing CSV
from pathlib import Path # For robust path handling
from sqlalchemy.ext.asyncio import AsyncSession
from decimal import Decimal
import structlog
from typing import Optional, Dict # Added Dict

from ..api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
from ..core.cache.cache_manager import CacheManager
from ..core.database.models.rvu_data_db import RVUDataModel
from sqlalchemy import select
from ..core.monitoring.app_metrics import MetricsCollector

logger = structlog.get_logger(__name__)

MEDICARE_CONVERSION_FACTOR = Decimal("38.87")
DEFAULT_RVU_VALUE = Decimal("1.00") # Default RVU if a procedure code is not found in DB

class RVUService:
    def __init__(self, cache_manager: CacheManager, metrics_collector: MetricsCollector):
        self.cache_manager = cache_manager
        self.metrics_collector = metrics_collector
        self.conversion_factor = MEDICARE_CONVERSION_FACTOR
        logger.info("RVUService initialized with CacheManager and MetricsCollector.",
                    cache_manager_id=id(cache_manager), metrics_collector_id=id(metrics_collector))

    async def _get_rvu_from_db(self, procedure_code: str, db_session: AsyncSession) -> Optional[Decimal]:
        """Fetches RVU for a single procedure code from the database."""
        try:
            async with self.metrics_collector.time_db_query("fetch_rvu_data_by_procedure_code"):
                stmt = select(RVUDataModel.rvu_value).where(RVUDataModel.procedure_code == procedure_code)
                result = await db_session.execute(stmt)
                rvu_value = result.scalar_one_or_none()
            if rvu_value is not None:
                logger.debug("RVU fetched from DB", procedure_code=procedure_code, rvu_value=rvu_value)
                return rvu_value
            else:
                logger.warn("RVU not found in DB for procedure code", procedure_code=procedure_code)
                return None
        except Exception as e:
            logger.error("Database error fetching RVU data", procedure_code=procedure_code, error=str(e))
            # Optionally, record a specific DB error metric here if desired
            return None

    async def calculate_rvu_for_claim(self, claim: ProcessableClaim, db_session: AsyncSession):
        logger.info("Calculating RVUs for claim (with cache and DB data)", claim_id=claim.claim_id, db_claim_id=claim.id)

        if not claim.line_items:
            logger.warn("Claim has no line items for RVU calculation", claim_id=claim.claim_id)
            return

        for line_item in claim.line_items:
            procedure_code = line_item.procedure_code
            units = line_item.units
            cache_key = f"rvu:{procedure_code}"
            rvu_per_unit: Optional[Decimal] = None

            # 1. Try to get from cache
            try:
                rvu_per_unit_str = await self.cache_manager.get(cache_key)
                if rvu_per_unit_str is not None:
                    try:
                        rvu_per_unit = Decimal(rvu_per_unit_str)
                        self.metrics_collector.record_cache_operation(cache_type='rvu', operation_type='get', outcome='hit')
                        logger.debug("RVU cache hit", procedure_code=procedure_code, rvu_value=rvu_per_unit)
                    except Exception as e:
                        logger.warn("Failed to parse cached RVU value, treating as miss.",
                                    procedure_code=procedure_code, cached_value=rvu_per_unit_str, error=str(e))
                        self.metrics_collector.record_cache_operation(cache_type='rvu', operation_type='get', outcome='parse_error')
                        rvu_per_unit = None # Force DB lookup
            except Exception as e: # Catch errors from cache_manager.get()
                logger.error("Cache get operation failed", cache_key=cache_key, error=str(e))
                self.metrics_collector.record_cache_operation(cache_type='rvu', operation_type='get', outcome='error')
                rvu_per_unit = None # Force DB lookup on cache error

            # 2. If cache miss or error, fetch from DB
            if rvu_per_unit is None:
                self.metrics_collector.record_cache_operation(cache_type='rvu', operation_type='get', outcome='miss') # Records miss if not already hit or error
                logger.debug("RVU cache miss, attempting DB lookup.", procedure_code=procedure_code)

                rvu_per_unit = await self._get_rvu_from_db(procedure_code, db_session)

                if rvu_per_unit is not None:
                    # Successfully fetched from DB, set in cache
                    try:
                        await self.cache_manager.set(cache_key, str(rvu_per_unit), ttl=3600)
                        # self.metrics_collector.record_cache_operation(cache_type='rvu', operation_type='set', outcome='success') # Implicitly covered by CacheManager
                    except Exception as e:
                        logger.error("Cache set operation failed after DB fetch", cache_key=cache_key, error=str(e))
                        # self.metrics_collector.record_cache_operation(cache_type='rvu', operation_type='set', outcome='error') # Implicitly covered by CacheManager
                else:
                    # Not found in DB, use default
                    logger.warn("RVU not found in DB, using default value.", procedure_code=procedure_code, default_rvu=DEFAULT_RVU_VALUE)
                    rvu_per_unit = DEFAULT_RVU_VALUE
                    # Optionally, cache the default value to prevent repeated DB lookups for unknown codes
                    try:
                        await self.cache_manager.set(cache_key, str(DEFAULT_RVU_VALUE), ttl=3600) # Cache default
                    except Exception as e:
                        logger.error("Cache set operation failed for default RVU", cache_key=cache_key, error=str(e))


            line_item.rvu_total = rvu_per_unit * Decimal(units)
            logger.debug(f"Line {line_item.line_number}: Proc {procedure_code}, Units {units}, RVU/Unit {rvu_per_unit}, Total RVU {line_item.rvu_total}")

        logger.info("Finished RVU calculation for claim (with cache and DB data)", claim_id=claim.claim_id, db_claim_id=claim.id)
