import pytest
import numpy as np
from datetime import date, datetime, timedelta # For creating ProcessableClaim
from decimal import Decimal # For creating ProcessableClaim

from claims_processor.src.processing.ml_pipeline.feature_extractor import FeatureExtractor
from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem # For test data
from claims_processor.src.core.config.settings import get_settings # To get configured feature count

@pytest.fixture
def feature_extractor() -> FeatureExtractor:
    # Settings will be loaded when FeatureExtractor is initialized
    return FeatureExtractor()

@pytest.fixture
def sample_processable_claim() -> ProcessableClaim:
    # Create a valid ProcessableClaim instance for testing
    # Most fields can be dummy but type-correct for the stub.
    line_item = ProcessableClaimLineItem(
        id=101, claim_db_id=1, line_number=1, service_date=date.today(),
        procedure_code="99213", units=1, charge_amount=Decimal("100.00"),
        created_at=datetime.now(), updated_at=datetime.now(), rvu_total=None
    )
    return ProcessableClaim(
        id=1, claim_id="ML_TEST_001", facility_id="F001", patient_account_number="P001",
        service_from_date=date.today(), service_to_date=date.today() + timedelta(days=1),
        total_charges=Decimal("100.00"), processing_status="pending",
        created_at=datetime.now(), updated_at=datetime.now(),
        line_items=[line_item],
        patient_first_name="Test", patient_last_name="User", patient_date_of_birth=date(1990,1,1)
    )

def test_extract_features_returns_correct_shape_and_type(
    feature_extractor: FeatureExtractor,
    sample_processable_claim: ProcessableClaim
):
    settings = get_settings()
    expected_feature_count = settings.ML_FEATURE_COUNT

    features = feature_extractor.extract_features(sample_processable_claim)

    assert isinstance(features, np.ndarray), "Features should be a NumPy array"
    assert features.shape == (1, expected_feature_count), \
        f"Expected shape (1, {expected_feature_count}), but got {features.shape}"
    assert features.dtype == np.float32, "Features should be float32 type"

    # Verify actual feature values based on sample_processable_claim
    # Expected features: total_charges, num_line_items, patient_age, service_duration_days, avg_charge_per_line, ...padding...

    # 1. Total Charges
    assert features[0, 0] == float(sample_processable_claim.total_charges)

    # 2. Number of Line Items
    assert features[0, 1] == float(len(sample_processable_claim.line_items))

    # 3. Patient Age
    expected_age = 0.0
    if sample_processable_claim.patient_date_of_birth:
        ref_date = sample_processable_claim.service_from_date if sample_processable_claim.service_from_date else date.today()
        expected_age = (ref_date - sample_processable_claim.patient_date_of_birth).days / 365.25
    assert np.isclose(features[0, 2], expected_age), f"Expected age {expected_age}, got {features[0, 2]}"

    # 4. Service Duration
    expected_duration = 0.0
    if sample_processable_claim.service_from_date and sample_processable_claim.service_to_date and \
       sample_processable_claim.service_to_date >= sample_processable_claim.service_from_date:
        expected_duration = float((sample_processable_claim.service_to_date - sample_processable_claim.service_from_date).days)
    assert np.isclose(features[0, 3], expected_duration), f"Expected duration {expected_duration}, got {features[0, 3]}"

    # 5. Average charge per line item
    expected_avg_charge = 0.0
    if sample_processable_claim.line_items: # Check if list is not empty
        total_line_charges = sum(float(li.charge_amount) for li in sample_processable_claim.line_items)
        if len(sample_processable_claim.line_items) > 0:
            expected_avg_charge = total_line_charges / len(sample_processable_claim.line_items)
    assert np.isclose(features[0, 4], expected_avg_charge), f"Expected avg charge {expected_avg_charge}, got {features[0, 4]}"

    # Check padding if feature_count > 5
    if expected_feature_count > 5:
        for i in range(5, expected_feature_count):
            assert features[0, i] == 0.0, f"Expected padding with 0.0 at feature index {i}, got {features[0,i]}"

def test_feature_extractor_initialization(feature_extractor: FeatureExtractor): # Renamed old test
    settings = get_settings()
    assert feature_extractor.feature_count == settings.ML_FEATURE_COUNT
    # Test that it logs, if caplog fixture is used (optional for now)
