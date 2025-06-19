import pytest
import numpy as np
from datetime import date, datetime, timedelta, timezone # Added timezone
from decimal import Decimal

from claims_processor.src.processing.ml_pipeline.feature_extractor import FeatureExtractor
from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
from claims_processor.src.core.config.settings import get_settings

@pytest.fixture
def feature_extractor() -> FeatureExtractor:
    return FeatureExtractor()

def create_test_claim(
    dob: Optional[date] = date(1990, 1, 1),
    service_from: date = date(2023, 1, 1),
    service_to: date = date(2023, 1, 3), # Results in 3 days duration (1, 2, 3)
    total_charges: Decimal = Decimal("100.00"),
    num_lines: int = 2,
    line_item_details: Optional[List[Dict[str, Any]]] = None # For more specific line items
) -> ProcessableClaim:
    lines = []
    if line_item_details:
        for i, details in enumerate(line_item_details):
            lines.append(ProcessableClaimLineItem(
                id=details.get("id", (i+1)*100),
                claim_db_id=details.get("claim_db_id", 1),
                line_number=details.get("line_number", i+1),
                service_date=details.get("service_date", service_from + timedelta(days=i)),
                procedure_code=details.get("procedure_code", f"P{i+1}"),
                units=details.get("units", 1),
                charge_amount=details.get("charge_amount", total_charges / (num_lines if num_lines > 0 else 1)),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                rvu_total=details.get("rvu_total")
            ))
    else:
        for i in range(num_lines):
            lines.append(ProcessableClaimLineItem(
                id=(i+1)*100, claim_db_id=1, line_number=i+1, service_date=service_from + timedelta(days=i),
                procedure_code=f"P{i+1}", units=1, charge_amount=total_charges / (num_lines if num_lines > 0 else 1),
                created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
            ))
    return ProcessableClaim(
        id=1, claim_id="test_claim_1", facility_id="fac1", patient_account_number="pa1",
        patient_date_of_birth=dob, service_from_date=service_from, service_to_date=service_to,
        total_charges=total_charges, processing_status="processing", created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc), line_items=lines
    )

def test_calculate_patient_age(feature_extractor: FeatureExtractor):
    dob = date(1990, 6, 15)
    ref_date_before_bday = date(2023, 6, 14)
    assert feature_extractor._calculate_patient_age(dob, ref_date_before_bday) == 32.0

    ref_date_on_bday = date(2023, 6, 15)
    assert feature_extractor._calculate_patient_age(dob, ref_date_on_bday) == 33.0

    ref_date_after_bday = date(2023, 6, 16)
    assert feature_extractor._calculate_patient_age(dob, ref_date_after_bday) == 33.0

    assert feature_extractor._calculate_patient_age(None, date.today()) is None
    # Test with invalid dob type (though Pydantic should catch this upstream)
    assert feature_extractor._calculate_patient_age("not-a-date", date.today()) is None


def test_calculate_service_duration(feature_extractor: FeatureExtractor):
    claim_same_day = create_test_claim(service_from=date(2023, 1, 1), service_to=date(2023, 1, 1))
    assert feature_extractor._calculate_service_duration(claim_same_day) == 1.0 # Same day is 1 day duration

    claim_multi_day = create_test_claim(service_from=date(2023, 1, 1), service_to=date(2023, 1, 3))
    assert feature_extractor._calculate_service_duration(claim_multi_day) == 3.0 # Jan 1, 2, 3 is 3 days

    claim_invalid_dates = create_test_claim(service_from=date(2023, 1, 3), service_to=date(2023, 1, 1))
    assert feature_extractor._calculate_service_duration(claim_invalid_dates) == 0.0 # Or specific error value

def test_normalize_total_charges(feature_extractor: FeatureExtractor):
    assert feature_extractor._normalize_total_charges(Decimal("123.45")) == 123.45
    assert feature_extractor._normalize_total_charges(0) == 0.0
    assert feature_extractor._normalize_total_charges("invalid") == 0.0 # Test conversion error

def test_encode_insurance_type(feature_extractor: FeatureExtractor):
    assert feature_extractor._encode_insurance_type("PPO") != 0.0 # Just check it does something
    assert feature_extractor._encode_insurance_type(None) == 0.0

def test_detect_surgery_codes(feature_extractor: FeatureExtractor):
    lines_with_surgery = [ProcessableClaimLineItem(id=1, claim_db_id=1, line_number=1, service_date=date.today(), procedure_code="SURG123", units=1, charge_amount=Decimal(10), created_at=datetime.now(),updated_at=datetime.now())]
    lines_without_surgery = [ProcessableClaimLineItem(id=1, claim_db_id=1, line_number=1, service_date=date.today(), procedure_code="P123", units=1, charge_amount=Decimal(10), created_at=datetime.now(),updated_at=datetime.now())]
    assert feature_extractor._detect_surgery_codes(lines_with_surgery) == 1.0
    assert feature_extractor._detect_surgery_codes(lines_without_surgery) == 0.0
    assert feature_extractor._detect_surgery_codes([]) == 0.0

def test_calculate_complexity_score(feature_extractor: FeatureExtractor):
    lines = [
        ProcessableClaimLineItem(id=1, claim_db_id=1, line_number=1, service_date=date.today(), procedure_code="P1", units=2, charge_amount=Decimal(10), created_at=datetime.now(),updated_at=datetime.now()),
        ProcessableClaimLineItem(id=2, claim_db_id=1, line_number=2, service_date=date.today(), procedure_code="P2", units=3, charge_amount=Decimal(10), created_at=datetime.now(),updated_at=datetime.now())
    ] # Total units = 5
    assert feature_extractor._calculate_complexity_score(lines) == 0.5 # 5/10.0
    assert feature_extractor._calculate_complexity_score([]) == 0.0
    lines_high_units = [ProcessableClaimLineItem(id=1, claim_db_id=1, line_number=1, service_date=date.today(), procedure_code="P1", units=15, charge_amount=Decimal(10), created_at=datetime.now(),updated_at=datetime.now())]
    assert feature_extractor._calculate_complexity_score(lines_high_units) == 1.0 # Capped at 10/10.0

def test_extract_features_logic_and_output(feature_extractor: FeatureExtractor):
    settings = get_settings() # FeatureExtractor itself doesn't use ML_FEATURE_COUNT from settings in the new version
    # The new FeatureExtractor always generates 7 features.
    # If settings.ML_FEATURE_COUNT is different, this test implicitly checks if the fixed output of 7 is an issue downstream.
    # For this unit test, we check against the fixed 7 features produced.
    expected_feature_count = 7

    test_dob = date(1980, 1, 15) # Approx 43 years before Jan 1, 2023
    test_service_from = date(2023, 1, 1)
    test_service_to = date(2023, 1, 3) # Duration 3 days
    test_total_charges = Decimal("250.75")
    test_num_lines = 2

    claim = create_test_claim(
        dob=test_dob,
        service_from=test_service_from,
        service_to=test_service_to,
        total_charges=test_total_charges,
        num_lines=test_num_lines
    )
    features = feature_extractor.extract_features(claim)

    assert isinstance(features, np.ndarray), "Features should be a NumPy array"
    assert features.shape == (1, expected_feature_count), \
        f"Expected shape (1, {expected_feature_count}), but got {features.shape}"
    assert features.dtype == np.float32, "Features should be float32 type"

    # 1. Total Charges
    assert np.isclose(features[0, 0], float(test_total_charges))
    # 2. Number of Line Items
    assert np.isclose(features[0, 1], float(test_num_lines))
    # 3. Patient Age
    expected_age = (test_service_from - test_dob).days / 365.25
    assert np.isclose(features[0, 2], expected_age)
    # 4. Service Duration
    expected_duration = float((test_service_to - test_service_from).days + 1)
    assert np.isclose(features[0, 3], expected_duration)
    # 5. Insurance Type Encoded (currently hardcoded to 0.0 as ProcessableClaim lacks insurance_type)
    assert np.isclose(features[0, 4], 0.0)
    # 6. Detect Surgery Codes (currently 0.0 based on placeholder logic and test data)
    assert np.isclose(features[0, 5], 0.0)
    # 7. Complexity Score (sum of units / 10.0, capped at 1.0)
    # Default units per line is 1. test_num_lines = 2. So sum is 2. Normalized is 2/10 = 0.2
    expected_complexity = min(float(test_num_lines * 1), 10.0) / 10.0
    assert np.isclose(features[0, 6], expected_complexity)


def test_extract_features_missing_dob(feature_extractor: FeatureExtractor):
    claim = create_test_claim(dob=None)
    features = feature_extractor.extract_features(claim)
    assert features is not None
    assert np.isclose(features[0, 2], -1.0) # Imputed age

def test_extract_features_handles_error_in_helper(feature_extractor: FeatureExtractor, monkeypatch):
    claim = create_test_claim()
    def mock_raise_error(*args, **kwargs):
        raise ValueError("Test error in helper")
    monkeypatch.setattr(feature_extractor, '_calculate_service_duration', mock_raise_error)
    assert feature_extractor.extract_features(claim) is None

def test_feature_extractor_initialization(feature_extractor: FeatureExtractor):
    # Simple test to ensure it initializes. Specific config loading could be tested if __init__ becomes complex.
    assert isinstance(feature_extractor, FeatureExtractor)
