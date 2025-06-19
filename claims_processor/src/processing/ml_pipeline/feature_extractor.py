from datetime import date, datetime
from typing import List, Optional, Any # Added Any for line_items in helper methods for now
# Assuming ProcessableClaim is correctly imported from its location
from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
import structlog
import numpy as np # For numerical operations and returning np.ndarray

logger = structlog.get_logger(__name__)

class FeatureExtractor:
    """
    Extracts ML features from claims data as per REQUIREMENTS.md.
    """

    def __init__(self):
        # In the future, this might load configuration for feature engineering,
        # e.g., encoding maps, scaling parameters, etc.
        # For this version, using feature_count from settings is handled by OptimizedPredictor/ClaimProcessingService,
        # this FeatureExtractor just produces a fixed set of 7 features as per plan.
        # If padding/truncation was needed, self.feature_count from settings would be used here.
        logger.info("FeatureExtractor initialized.")

    def _calculate_patient_age(self, patient_date_of_birth: Optional[date], reference_date: Optional[date] = None) -> Optional[float]:
        """Calculates patient age in years as of the reference date (or today)."""
        if not patient_date_of_birth:
            return None

        # Use claim's service_from_date as reference if no explicit reference_date
        effective_reference_date = reference_date if reference_date else date.today()

        try:
            # Ensure patient_date_of_birth is a date object
            if not isinstance(patient_date_of_birth, date):
                logger.warn("patient_date_of_birth is not a valid date object", dob_type=type(patient_date_of_birth))
                return None # Or attempt to parse if it's a string, but Pydantic should handle this.

            age = effective_reference_date.year - patient_date_of_birth.year - \
                  ((effective_reference_date.month, effective_reference_date.day) <
                   (patient_date_of_birth.month, patient_date_of_birth.day))
            return float(age)
        except Exception as e:
            logger.warn("Could not calculate patient age", dob=patient_date_of_birth, ref_date=effective_reference_date, error=str(e))
            return None

    def _calculate_service_duration(self, claim: ProcessableClaim) -> float:
        """Calculates the duration of service in days."""
        if not claim.service_from_date or not claim.service_to_date or claim.service_to_date < claim.service_from_date:
            return 0.0 # Or a specific value indicating invalid/unknown duration
        duration = (claim.service_to_date - claim.service_from_date).days
        return float(duration + 1)

    def _normalize_total_charges(self, total_charges: Any, default_max_charge: float = 100000.0) -> float:
        """ Normalizes total charges. Using raw float value for now."""
        try:
            val = float(total_charges) # total_charges is Decimal from Pydantic model
            return val
        except (ValueError, TypeError):
            logger.warn("Could not convert total_charges to float for normalization", value=total_charges)
            return 0.0


    def _encode_insurance_type(self, insurance_type: Optional[str]) -> float:
        """ Encodes insurance type. Placeholder logic."""
        if insurance_type:
            # Simple placeholder. Real implementation needs a fixed mapping.
            return float(abs(hash(insurance_type.lower())) % 100) / 100.0
        return 0.0

    def _detect_surgery_codes(self, line_items: List[ProcessableClaimLineItem]) -> float:
        """ Detects if any surgery codes are present. Placeholder logic."""
        if not line_items: return 0.0
        for item in line_items:
            # Extremely naive placeholder. Real logic would check procedure codes against known surgical ranges/lists.
            if item.procedure_code and ("SURG" in item.procedure_code.upper() or item.procedure_code.startswith("S")): # Example
                 return 1.0
        return 0.0

    def _calculate_complexity_score(self, line_items: List[ProcessableClaimLineItem]) -> float:
        """ Calculates a complexity score based on line items. Placeholder logic."""
        if not line_items: return 0.0
        score = 0.0
        for item in line_items:
            score += item.units # Example: more units = more complex
        # Normalize score, e.g. assuming max 10 units sum is "complex"
        return min(score, 10.0) / 10.0

    def extract_features(self, claim: ProcessableClaim) -> Optional[np.ndarray]:
        """
        Extracts a defined set of 7 features from a ProcessableClaim.
        Returns a NumPy array of features (shape (7,)), or None if essential data is missing leading to error.
        Order of features: total_charges, line_item_count, patient_age, service_duration_days,
                          insurance_type_encoded, detect_surgery_codes, procedure_complexity_score.
        """
        logger.debug("Extracting features for claim", claim_id=claim.claim_id)
        try:
            patient_age_val = self._calculate_patient_age(claim.patient_date_of_birth, reference_date=claim.service_from_date)

            # ProcessableClaim does not have insurance_type. Passing None.
            # This feature will be 0.0 unless ProcessableClaim is updated or data sourced differently.
            insurance_type_encoded_val = self._encode_insurance_type(getattr(claim, 'insurance_type', None))

            features = [
                self._normalize_total_charges(claim.total_charges),
                float(len(claim.line_items)) if claim.line_items else 0.0,
                patient_age_val if patient_age_val is not None else -1.0, # Impute missing age with -1 or other strategy
                self._calculate_service_duration(claim),
                insurance_type_encoded_val,
                self._detect_surgery_codes(claim.line_items),
                self._calculate_complexity_score(claim.line_items)
            ]

            # Convert to NumPy array with expected dtype (float32) and shape (1, 7)
            # The previous feature extractor (subtask 21) padded to settings.ML_FEATURE_COUNT.
            # This version produces a fixed set of 7 features as per the plan's example list.
            # If ML_FEATURE_COUNT in settings is different from 7, this would need adjustment or padding.
            # For now, assuming 7 features are always generated.
            features_array = np.array(features, dtype=np.float32)

            # The ML model might expect shape (1, num_features) for a single prediction.
            # Reshape if necessary based on model input specs.
            # The OptimizedPredictor stub expects (num_samples, num_features).
            # So, if this is for one claim, it should be (1, 7).
            if features_array.ndim == 1:
                 features_array = features_array.reshape(1, -1)


            logger.info(
                "Features extracted for claim",
                claim_id=claim.claim_id,
                shape=features_array.shape,
                num_features=features_array.shape[1],
                features_preview=features_array[0, :min(features_array.shape[1], 7)] # Log first few actual features
            )
            return features_array

        except Exception as e:
            logger.error("Failed to extract features for claim", claim_id=claim.claim_id, error=str(e), exc_info=True)
            return None # Return None or an array of default values (e.g., zeros)
                                # depending on how downstream ML model handles missing feature sets.
                                # For now, None, and predictor should handle None input if necessary,
                                # though current predictor expects ndarray.
                                # The ClaimProcessingService will skip ML if features are None.
                                # Let's return an empty array of expected shape to avoid breaking OptimizedPredictor input type.
                                # Or, ensure OptimizedPredictor handles None input.
                                # The plan for ClaimProcessingService was to skip ML if features are None.
                                # So returning None is fine.

                                # Actually, OptimizedPredictor expects np.ndarray.
                                # Let's return a default array of zeros if extraction fails,
                                # matching the feature_count specified in settings.
                                # This requires getting settings here or assuming a fixed count.
                                # The previous version of FeatureExtractor used self.feature_count from settings.
                                # This one doesn't store it in __init__.
                                # Let's re-add it to __init__ for consistency.
                                # For now, this error path returns None.
                                # The calling service (ClaimProcessingService) will check for None.
            return None
