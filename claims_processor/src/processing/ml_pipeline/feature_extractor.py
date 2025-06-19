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
    # Add more as needed
}
DEFAULT_INSURANCE_ENCODING = 0.0 # For 'other' or None

SURGERY_CODE_PREFIXES = {"S", "SURG", "10", "11"} # Example prefixes. Actual CPT ranges for surgery are 10000-69999.

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
            if item.procedure_code:
                # More realistic check against CPT surgery ranges (10000-69999)
                # This requires procedure_code to be clean numeric strings for range checks.
                # The current prefixes are just illustrative.
                try:
                    # Attempt to check if it's a number first
                    proc_code_num = int(item.procedure_code)
                    if 10000 <= proc_code_num <= 69999:
                        return 1.0
                except ValueError:
                    # If not a number, check against string prefixes (as in original placeholder)
                    if any(item.procedure_code.upper().startswith(prefix) for prefix in SURGERY_CODE_PREFIXES):
                        return 1.0
        return 0.0

    def _calculate_complexity_score(self, line_items: List[ProcessableClaimLineItem]) -> float:
        if not line_items: return 0.0
        score = 0.0
        num_lines = len(line_items)
        total_units = sum(item.units for item in line_items if item.units is not None) # item.units is int, not Optional

        score += min(num_lines / 10.0, 0.5)
        score += min(total_units / 20.0, 0.5)
        return min(score, 1.0)

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

            # Ensure 7 features are always produced. If fewer, this indicates an issue or need for padding.
            # The current implementation generates exactly 7 features.
            # If settings.ML_FEATURE_COUNT was different, padding/truncation logic would be needed here.
            if len(features) != 7: # Fixed number of features
                 logger.error(f"Feature extraction resulted in {len(features)} features, expected 7.", claim_id=claim.claim_id)
                 # Decide error handling: return None, or pad/truncate. For now, log and return as is.
                 # This would ideally align with a self.feature_count from settings if used.

            features_array = np.array(features, dtype=np.float32) # This creates a 1D array (7,)

            # The OptimizedPredictor.predict_batch expects a list of features, where each feature set
            # is a 1D np.ndarray. The old FeatureExtractor returned (1,N).
            # The current OptimizedPredictor's predict_batch has logic to reshape (1,N) to (N,).
            # To be safe and explicit, let's ensure this returns (1,N) as previously,
            # as OptimizedPredictor.predict_batch expects a list of these.
            # The _apply_ml_predictions method in ParallelClaimsProcessor collects these (1,N) arrays
            # into a list features_batch_for_predictor: List[np.ndarray].
            # Then OptimizedPredictor.predict_batch receives this list. Its internal loop processes
            # each item. If item is (1,N), it reshapes.
            # So, (1,N) output from here is consistent with current OptimizedPredictor.
            if features_array.ndim == 1:
                 features_array = features_array.reshape(1, -1)


            logger.info(
                "Features extracted for claim",
                claim_id=claim.claim_id,
                shape=features_array.shape,
                features_preview=features_array[0, :min(features_array.shape[1], 7)]
            )
            return features_array

        except Exception as e:
            logger.error("Failed to extract features for claim", claim_id=claim.claim_id, error=str(e), exc_info=True)
            # Return a default array of zeros matching the expected shape (1, num_features)
            # This requires knowing num_features. If FeatureExtractor had self.feature_count from settings:
            # from claims_processor.src.core.config.settings import get_settings # Temporary if not in __init__
            # settings = get_settings()
            # default_features = np.zeros((1, settings.ML_FEATURE_COUNT), dtype=np.float32)
            # logger.warn("Returning default features due to error", claim_id=claim.claim_id, shape=default_features.shape)
            # return default_features
            # For now, returning None as per previous design, and caller handles None.
            return None
