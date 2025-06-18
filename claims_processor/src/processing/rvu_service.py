from sqlalchemy.ext.asyncio import AsyncSession
from decimal import Decimal
import structlog
from typing import Optional # Added for rvu_per_unit type hint

from ..api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
from ..core.cache.cache_manager import CacheManager # Import CacheManager

logger = structlog.get_logger(__name__)

MOCK_RVU_DATA = {
    "99213": Decimal("2.11"),
    "99214": Decimal("3.28"),
    "80053": Decimal("0.85"),
    "DEFAULT_RVU": Decimal("1.00")
}
MEDICARE_CONVERSION_FACTOR = Decimal("38.87")

class RVUService:
    def __init__(self, cache_manager: CacheManager): # Accept CacheManager
        self.rvu_data = MOCK_RVU_DATA
        self.conversion_factor = MEDICARE_CONVERSION_FACTOR
        self.cache_manager = cache_manager # Store cache manager
        logger.info("RVUService initialized with CacheManager and mock data.")

    async def calculate_rvu_for_claim(self, claim: ProcessableClaim, db_session: AsyncSession): # db_session might not be used if RVUs are self-contained
        logger.info("Calculating RVUs for claim (with cache)", claim_id=claim.claim_id, db_claim_id=claim.id)

        if not claim.line_items:
            logger.warn("Claim has no line items for RVU calculation", claim_id=claim.claim_id)
            return

        for line_item in claim.line_items:
            procedure_code = line_item.procedure_code
            # units are int as per ProcessableClaimLineItem model, direct use is fine for multiplication with Decimal
            units = line_item.units

            cache_key = f"rvu:{procedure_code}"
            rvu_per_unit_str = await self.cache_manager.get(cache_key)

            rvu_per_unit: Optional[Decimal] = None

            if rvu_per_unit_str is not None:
                try:
                    rvu_per_unit = Decimal(rvu_per_unit_str)
                    logger.debug("RVU cache hit", procedure_code=procedure_code, rvu_value=rvu_per_unit)
                except Exception as e:
                    logger.warn("Failed to parse cached RVU value, falling back to mock data.",
                                procedure_code=procedure_code, cached_value=rvu_per_unit_str, error=str(e))
                    # Fallback to mock data if parsing fails

            if rvu_per_unit is None: # Cache miss or parsing failed
                logger.debug("RVU cache miss or parse error, using mock data.", procedure_code=procedure_code)
                rvu_per_unit = self.rvu_data.get(procedure_code, self.rvu_data["DEFAULT_RVU"])
                # Store in cache for next time
                await self.cache_manager.set(cache_key, str(rvu_per_unit), ttl=3600) # Store as string

            line_item.rvu_total = rvu_per_unit * Decimal(units) # Ensure units is also Decimal for precision if it could be float

            logger.debug(f"Line {line_item.line_number}: Proc {procedure_code}, Units {units}, RVU/Unit {rvu_per_unit}, Total RVU {line_item.rvu_total}")
            # Future: Calculate and store line_item.expected_reimbursement if model supports it
            # line_expected_reimbursement = line_item.rvu_total * self.conversion_factor
            # line_item.expected_reimbursement = line_expected_reimbursement.quantize(Decimal("0.01"))

        # Future: Calculate and store claim.expected_total_reimbursement if model supports it
        logger.info("Finished RVU calculation for claim (with cache)", claim_id=claim.claim_id, db_claim_id=claim.id)
