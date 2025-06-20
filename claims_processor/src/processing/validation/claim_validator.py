from datetime import date
from typing import List, Dict, Any
# Assuming ProcessableClaim and ProcessableClaimLineItem are in:
from ...api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
import structlog

logger = structlog.get_logger(__name__)

class ClaimValidator:
    def __init__(self):
        # Potentially load more complex validation rules from config/DB later
        pass

    def validate_claim(self, claim: ProcessableClaim) -> List[str]:
        """
        Validates a single claim based on a set of predefined rules.
        Returns a list of error messages. An empty list means the claim is valid.
        """
        errors: List[str] = []

        # Basic presence checks for claim_id, facility_id, patient_account_number
        # are now primarily handled by Pydantic model validation (constr with min_length=1)
        # if ProcessableClaim inherits or uses such constraints.
        # If ProcessableClaim allows empty strings for these, then these checks might still be relevant.
        # Assuming Pydantic model ensures these are non-empty if they are not Optional.

        # Rule: service_from_date <= service_to_date
        if claim.service_from_date > claim.service_to_date:
            errors.append(f"Service-from date ({claim.service_from_date}) cannot be after service-to date ({claim.service_to_date}).")

        # Rule: total_charges > 0 (This is now handled by Pydantic `gt=Decimal(0)` on ProcessableClaim.total_charges)
        # if claim.total_charges <= 0:
        #     errors.append(f"Total charges ({claim.total_charges}) must be positive.")

        # Rule: Line items must not be empty (Handled by Pydantic `min_length=1` if ProcessableClaim.line_items uses it)
        # For ProcessableClaim, line_items: List[ProcessableClaimLineItem] = [] is default, so a check here is still useful.
        if not claim.line_items:
            errors.append("Claim must have at least one line item.")
        else:
            # Validate each line item
            for i, line_item in enumerate(claim.line_items):
                line_errors = self._validate_line_item(line_item, line_number=i + 1, claim_service_from_date=claim.service_from_date, claim_service_to_date=claim.service_to_date)
                errors.extend(line_errors)

            # Rule: Sum of line item charges should ideally match total_charges
            # This check is only performed if there are no line item errors that might affect charge_amount.
            # However, individual line item charge_amount validation (e.g. non-negative) is in _validate_line_item.
            # Pydantic on IngestionClaimLineItem now has ge=Decimal(0) for charge_amount.
            # ProcessableClaimLineItem also has condecimal for charge_amount.
            current_total_line_charges = sum(li.charge_amount for li in claim.line_items)
            if current_total_line_charges != claim.total_charges:
                errors.append(
                    f"Sum of line item charges ({current_total_line_charges}) does not match claim total charges ({claim.total_charges})."
                )

        # Add more validation rules as per REQUIREMENTS.md here
        # e.g., financial_class, insurance_type, provider NPI formats (regex), diagnosis codes format etc.

        if errors:
            logger.debug("Claim validation failed", claim_id=claim.claim_id, errors=errors)
        else:
            logger.debug("Claim validation successful", claim_id=claim.claim_id)

        return errors

    def _validate_line_item(self, line_item: ProcessableClaimLineItem, line_number: int, claim_service_from_date: date, claim_service_to_date: date) -> List[str]:
        """Validates a single claim line item."""
        line_errors: List[str] = []

        if not line_item.procedure_code:
            line_errors.append(f"Line {line_number}: Missing procedure_code.")

        if line_item.units <= 0:
            line_errors.append(f"Line {line_number}: Units ({line_item.units}) must be positive.")

        if line_item.charge_amount < 0: # Allow zero charge for some line items? REQUIREMENTS.md says positive_charge for claim_line_items_template >= 0
            line_errors.append(f"Line {line_number}: Charge amount ({line_item.charge_amount}) cannot be negative.")

        # Rule: service_date of line item should be within claim's service_from_date and service_to_date
        if not (claim_service_from_date <= line_item.service_date <= claim_service_to_date):
            line_errors.append(f"Line {line_number}: Service date ({line_item.service_date}) is outside of claim service period ({claim_service_from_date} to {claim_service_to_date}).")

        return line_errors
