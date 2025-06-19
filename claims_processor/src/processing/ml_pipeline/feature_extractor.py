import numpy as np
import structlog
from typing import Dict, List, Optional # Added List, Optional for conceptual methods
from datetime import date # Added for conceptual methods

# Assuming ProcessableClaim is in:
from ....api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem # Added ProcessableClaimLineItem for conceptual
from ....core.config.settings import get_settings # To get ML_FEATURE_COUNT

logger = structlog.get_logger(__name__)

class FeatureExtractor:
    def __init__(self):
        settings = get_settings()
        self.feature_count = settings.ML_FEATURE_COUNT
        logger.info("FeatureExtractor initialized.", feature_count=self.feature_count)

    def extract_features(self, claim: ProcessableClaim) -> np.ndarray:
        logger.debug("Extracting features for claim", claim_id=claim.claim_id, db_id=claim.id)

        feature_list = []

        # 1. Total Charges
        feature_list.append(float(claim.total_charges))

        # 2. Number of Line Items
        feature_list.append(float(len(claim.line_items)))

        # 3. Patient Age (Years) at time of service
        age_in_years: float = 0.0
        if claim.patient_date_of_birth:
            reference_date = claim.service_from_date if claim.service_from_date else date.today()
            age_in_years = (reference_date - claim.patient_date_of_birth).days / 365.25
        feature_list.append(age_in_years)

        # 4. Service Duration (Days)
        service_duration_days: float = 0.0
        if claim.service_from_date and claim.service_to_date and claim.service_to_date >= claim.service_from_date:
            duration = claim.service_to_date - claim.service_from_date
            service_duration_days = float(duration.days)
        feature_list.append(service_duration_days)

        # 5. Average charge per line item
        avg_charge_per_line: float = 0.0
        if claim.line_items: # Check if list is not empty
            total_line_charges = sum(float(li.charge_amount) for li in claim.line_items)
            avg_charge_per_line = total_line_charges / len(claim.line_items) if len(claim.line_items) > 0 else 0.0
        feature_list.append(avg_charge_per_line)

        # Pad with zeros if fewer features are implemented than self.feature_count
        num_implemented_features = len(feature_list)
        if num_implemented_features < self.feature_count:
            padding = [0.0] * (self.feature_count - num_implemented_features)
            feature_list.extend(padding)
        elif num_implemented_features > self.feature_count:
            feature_list = feature_list[:self.feature_count] # Truncate

        features_array = np.array([feature_list], dtype=np.float32)

        logger.info(
            "Features extracted for claim",
            claim_id=claim.claim_id,
            shape=features_array.shape,
            num_features_implemented=num_implemented_features,
            features_preview=features_array[:, :min(num_implemented_features, 5)]
        )
        return features_array

    # Example helper methods (conceptual) - can be removed or kept commented
    # def _encode_insurance_type(self, insurance_type: Optional[str]) -> int:
    #     # Simple encoding logic
    #     if not insurance_type: return 0
    #     if "PPO" in insurance_type.upper(): return 1
    #     if "HMO" in insurance_type.upper(): return 2
    #     return 3 # Other

    # def _calculate_complexity_score(self, line_items: List[ProcessableClaimLineItem]) -> float:
    #     # Example: sum of (units * charge_amount / 1000) for each line
    #     score = 0.0
    #     for item in line_items:
    #         score += (item.units * float(item.charge_amount)) / 1000.0 # Ensure float arithmetic
    #     return score
