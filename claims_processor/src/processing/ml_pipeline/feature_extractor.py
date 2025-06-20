from datetime import date, datetime # Keep datetime if ProcessableClaim has datetime fields
from typing import List, Optional, Any
from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
import structlog
import numpy as np
from decimal import Decimal # Import Decimal for type consistency if needed, though helpers return float

logger = structlog.get_logger(__name__)

# Define constants for feature encoding at module or class level
INSURANCE_TYPE_MAPPING = {
    "medicare": 1.0,
    "medicaid": 2.0,
    "commercial": 3.0,
    "self-pay": 4.0,
    "hmo": 5.0,
    "ppo": 6.0,
    "other": 7.0,
    # Add more as needed
}
DEFAULT_INSURANCE_ENCODING = 0.0 # For unknown, None, or unmapped 'other'

# Define CPT code ranges for surgery (inclusive)
# General Surgery: 10021–69990. This is a broad range.
# Specific sub-ranges can be added for more granularity if needed.
SURGERY_CPT_RANGES = [
    (10021, 69990),
    # Example: Add other surgery-like code ranges if necessary, e.g., from HCPCS level II
]
# HCPCS Level II "S" codes are surgical procedures, but can also be non-surgical.
# For simplicity, we'll focus on CPT ranges and a few prefixes.
# If a code starts with 'S' and is not in CPT format, it might be a HCPCS Level II code.
# Let's keep a small set of prefixes for non-numeric or non-CPT codes if needed.
NON_NUMERIC_SURGERY_INDICATORS = {"S"} # Example: HCPCS Level II 'S' codes often indicate surgery

class FeatureExtractor:
    """
    Extracts ML features from claims data.
    """

    def __init__(self):
        # In a real scenario, settings might provide feature_count, normalization params, etc.
        # For this implementation, it produces a fixed set of 7 features.
        # self.feature_count = 7 # Or get from settings if padding/truncation logic is re-added.
        logger.info("FeatureExtractor initialized.")

    def _calculate_patient_age(self, patient_date_of_birth: Optional[date], reference_date: Optional[date] = None) -> Optional[float]:
        if not patient_date_of_birth:
            return None

        effective_reference_date = reference_date if reference_date else date.today()

        try:
            if not isinstance(patient_date_of_birth, date): # Should be handled by Pydantic, but good check
                logger.warn("patient_date_of_birth is not a valid date object", dob_type=type(patient_date_of_birth))
                return None

            age = effective_reference_date.year - patient_date_of_birth.year - \
                  ((effective_reference_date.month, effective_reference_date.day) <
                   (patient_date_of_birth.month, patient_date_of_birth.day))
            return float(age)
        except Exception as e:
            logger.warn("Could not calculate patient age", dob=patient_date_of_birth, ref_date=effective_reference_date, error=str(e))
            return None

    def _calculate_service_duration(self, claim: ProcessableClaim) -> float:
        if not claim.service_from_date or not claim.service_to_date or claim.service_to_date < claim.service_from_date:
            return 0.0
        duration = (claim.service_to_date - claim.service_from_date).days
        return float(duration + 1)

    def _normalize_total_charges(self, total_charges: Decimal) -> float: # Takes Decimal from Pydantic
        # Using log1p for normalization, handles 0, avoids log(0)
        # Ensure total_charges is float for np.log1p
        charge_float = float(total_charges)
        return np.log1p(charge_float if charge_float > 0 else 0.0)

    def _encode_insurance_type(self, insurance_type: Optional[str]) -> float:
        if insurance_type:
            return INSURANCE_TYPE_MAPPING.get(insurance_type.lower().strip(), DEFAULT_INSURANCE_ENCODING)
        return DEFAULT_INSURANCE_ENCODING

    def _detect_surgery_codes(self, line_items: List[ProcessableClaimLineItem]) -> float:
        if not line_items: return 0.0
        for item in line_items:
            proc_code = item.procedure_code.strip().upper() if item.procedure_code else ""
            if not proc_code:
                continue

            # Check CPT ranges
            try:
                # Handle cases where proc_code might have non-numeric characters but starts numeric for CPTs
                # e.g. "12345A". For strict CPT, it should be purely numeric for these ranges.
                # For simplicity, assume CPT codes are passed as clean numeric strings if they are CPTs.
                # If a code like "1234X" should be invalid, more parsing is needed.
                # This simplified try-except handles if item.procedure_code is not purely int-convertible.
                numeric_part_of_code = ""
                for char_code in proc_code:
                    if char_code.isdigit():
                        numeric_part_of_code += char_code
                    else: # Stop at first non-digit for range check if style is e.g. 12345A
                        break

                if numeric_part_of_code: # If there was any numeric part
                    code_num = int(numeric_part_of_code)
                    for r_start, r_end in SURGERY_CPT_RANGES:
                        if r_start <= code_num <= r_end:
                            return 1.0
            except ValueError:
                # Not a purely numeric code or numeric prefix, check string indicators
                pass # Fall through to string prefix checks

            # Check for non-numeric indicators (e.g. HCPCS 'S' codes)
            if any(proc_code.startswith(indicator) for indicator in NON_NUMERIC_SURGERY_INDICATORS):
                return 1.0
        return 0.0

    def _calculate_complexity_score(self, claim: ProcessableClaim, surgery_detected_flag: float) -> float:
        # claim argument is now the full ProcessableClaim
        line_items = claim.line_items
        if not line_items: return 0.0

        num_lines = len(line_items)
        total_units = sum(item.units for item in line_items) # item.units is int, not Optional

        # Base score components
        lines_score = min(num_lines / 10.0, 0.4)  # Max 0.4 from number of lines (e.g., 10+ lines)
        units_score = min(total_units / 20.0, 0.3) # Max 0.3 from total units (e.g., 20+ units)
        surgery_score = 0.3 if surgery_detected_flag == 1.0 else 0.0 # 0.3 if surgery

        # Combine scores
        # Example: Normalize to 0-1 range. Max possible sum here is 0.4 + 0.3 + 0.3 = 1.0
        complexity = lines_score + units_score + surgery_score

        return min(max(complexity, 0.0), 1.0) # Ensure it's capped between 0 and 1


    def extract_features(self, claim: ProcessableClaim) -> Optional[np.ndarray]:
        logger.debug("Extracting features for claim", claim_id=claim.claim_id)
        try:
            patient_age_val = self._calculate_patient_age(claim.patient_date_of_birth, reference_date=claim.service_from_date)

            # Use the new insurance_type field from ProcessableClaim
            insurance_type_encoded_val = self._encode_insurance_type(claim.insurance_type)

            features = [
                self._normalize_total_charges(claim.total_charges),
                float(len(claim.line_items)) if claim.line_items else 0.0,
                patient_age_val if patient_age_val is not None else -1.0,
                self._calculate_service_duration(claim),
                insurance_type_encoded_val,
                self._detect_surgery_codes(claim.line_items),
                self._calculate_complexity_score(claim.line_items)
            ]

            patient_age_val = self._calculate_patient_age(claim.patient_date_of_birth, reference_date=claim.service_from_date)
            insurance_type_encoded_val = self._encode_insurance_type(claim.insurance_type)
            surgery_detected_flag = self._detect_surgery_codes(claim.line_items)
            complexity_score_val = self._calculate_complexity_score(claim, surgery_detected_flag) # Pass full claim

            features = [
                self._normalize_total_charges(claim.total_charges),
                float(len(claim.line_items)) if claim.line_items else 0.0,
                patient_age_val if patient_age_val is not None else -1.0, # Use -1.0 for missing age as a common practice
                self._calculate_service_duration(claim),
                insurance_type_encoded_val,
                surgery_detected_flag,
                complexity_score_val
            ]

            # Ensure 7 features are always produced.
            expected_feature_count = 7
            if len(features) != expected_feature_count:
                 logger.error(f"Feature extraction resulted in {len(features)} features, expected {expected_feature_count}.",
                              claim_id=claim.claim_id)
                 # This case should ideally not be hit if all helpers provide a default value.
                 # If it is hit, returning None or raising an error might be options.
                 # For now, returning None as per original error handling for critical failures.
                 return None

            # Output should be a 1D NumPy array of shape (feature_count,)
            features_array = np.array(features, dtype=np.float32)

            # Ensure it's 1D. If somehow it became 2D (e.g. [[...]]), take the first row.
            if features_array.ndim > 1 and features_array.shape[0] == 1:
                features_array = features_array.reshape(-1) # Reshape (1,N) to (N,)

            if features_array.shape[0] != expected_feature_count: # Final check on dimension
                 logger.error(f"Final features_array shape is {features_array.shape}, expected ({expected_feature_count},).",
                              claim_id=claim.claim_id)
                 return None


            logger.info(
                "Features extracted for claim",
                claim_id=claim.claim_id,
                shape=str(features_array.shape), # str() for logging tuple
                features_preview=str(features_array[:min(len(features_array), 7)]) # str() for logging array slice
            )
            return features_array

        except Exception as e:
            logger.error("Failed to extract features for claim", claim_id=claim.claim_id, error=str(e), exc_info=True)
            return None
