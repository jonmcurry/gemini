import pytest
import numpy as np
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Optional, Any # Added for create_test_claim

from claims_processor.src.processing.ml_pipeline.feature_extractor import FeatureExtractor, INSURANCE_TYPE_MAPPING, DEFAULT_INSURANCE_ENCODING, SURGERY_CODE_PREFIXES
from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
from claims_processor.src.core.config.settings import get_settings

@pytest.fixture
def feature_extractor() -> FeatureExtractor:
    return FeatureExtractor()

# Updated create_test_claim helper
def create_test_claim(
    dob: Optional[date] = date(1990, 1, 1),
    service_from: date = date(2023, 1, 1),
    service_to: date = date(2023, 1, 3),
    total_charges: Decimal = Decimal("100.00"),
    num_lines: int = 2,
    line_item_details: Optional[List[Dict[str, Any]]] = None,
    insurance_type: Optional[str] = "commercial", # Added
    financial_class: Optional[str] = "FC1"      # Added
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
        for i in range(num_lines): # Ensure num_lines is respected
            lines.append(ProcessableClaimLineItem(
                id=(i+1)*100, claim_db_id=1, line_number=i+1,
                service_date=service_from + timedelta(days=i if i < (service_to - service_from).days +1 else 0), # Ensure line service date is within claim dates
                procedure_code=f"P{i+1}", units=1,
                charge_amount=total_charges / (num_lines if num_lines > 0 else 1),
                created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
            ))
    return ProcessableClaim(
        id=1, claim_id="test_claim_1", facility_id="fac1", patient_account_number="pa1",
        patient_date_of_birth=dob, service_from_date=service_from, service_to_date=service_to,
        total_charges=total_charges, processing_status="processing", created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc), line_items=lines,
        insurance_type=insurance_type, financial_class=financial_class # Added
    )

# --- Tests for Helper Methods ---
def test_calculate_patient_age_detailed(feature_extractor: FeatureExtractor):
    dob = date(1990, 6, 15)
    assert feature_extractor._calculate_patient_age(dob, date(2023, 6, 14)) == 32.0
    assert feature_extractor._calculate_patient_age(dob, date(2023, 6, 15)) == 33.0
    assert feature_extractor._calculate_patient_age(dob, date(2023, 6, 16)) == 33.0
    assert feature_extractor._calculate_patient_age(None, date.today()) is None
    assert feature_extractor._calculate_patient_age("not-a-date", date.today()) is None # type: ignore

def test_calculate_service_duration_detailed(feature_extractor: FeatureExtractor):
    claim_same_day = create_test_claim(service_from=date(2023, 1, 1), service_to=date(2023, 1, 1))
    assert feature_extractor._calculate_service_duration(claim_same_day) == 1.0
    claim_multi_day = create_test_claim(service_from=date(2023, 1, 1), service_to=date(2023, 1, 3))
    assert feature_extractor._calculate_service_duration(claim_multi_day) == 3.0
    claim_invalid_dates = create_test_claim(service_from=date(2023, 1, 3), service_to=date(2023, 1, 1))
    assert feature_extractor._calculate_service_duration(claim_invalid_dates) == 0.0
    claim_no_from_date = create_test_claim(service_from=None, service_to=date(2023,1,1)) # type: ignore
    assert feature_extractor._calculate_service_duration(claim_no_from_date) == 0.0
    claim_no_to_date = create_test_claim(service_from=date(2023,1,1), service_to=None) # type: ignore
    assert feature_extractor._calculate_service_duration(claim_no_to_date) == 0.0


def test_normalize_total_charges_detailed(feature_extractor: FeatureExtractor):
    assert feature_extractor._normalize_total_charges(Decimal("0.0")) == np.log1p(0.0)
    assert feature_extractor._normalize_total_charges(Decimal("100.0")) == np.log1p(100.0)
    assert feature_extractor._normalize_total_charges(Decimal("99999.0")) == np.log1p(99999.0)
    # Test with float input, though Pydantic model uses Decimal
    assert feature_extractor._normalize_total_charges(100.0) == np.log1p(100.0) # type: ignore
    # The method expects Decimal from Pydantic, but test with float for robustness of float() cast
    with pytest.raises(TypeError): # Or handle gracefully if that's the design
         feature_extractor._normalize_total_charges("invalid_type") # type: ignore


def test_encode_insurance_type_detailed(feature_extractor: FeatureExtractor):
    assert feature_extractor._encode_insurance_type("Medicare") == INSURANCE_TYPE_MAPPING.get("medicare")
    assert feature_extractor._encode_insurance_type(" medicaid ") == INSURANCE_TYPE_MAPPING.get("medicaid") # Test stripping
    assert feature_extractor._encode_insurance_type("Commercial") == INSURANCE_TYPE_MAPPING.get("commercial")
    assert feature_extractor._encode_insurance_type("SELF-PAY") == INSURANCE_TYPE_MAPPING.get("self-pay")
    assert feature_extractor._encode_insurance_type("UnknownType") == DEFAULT_INSURANCE_ENCODING
    assert feature_extractor._encode_insurance_type(None) == DEFAULT_INSURANCE_ENCODING

def test_detect_surgery_codes_detailed(feature_extractor: FeatureExtractor):
    now = datetime.now(timezone.utc)
    # Example CPT codes for surgery range from 10000-69999
    line_surg_cpt_range_start = ProcessableClaimLineItem(id=1, claim_db_id=1, line_number=1, service_date=date.today(), procedure_code="10021", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now)
    line_surg_cpt_range_end = ProcessableClaimLineItem(id=2, claim_db_id=1, line_number=2, service_date=date.today(), procedure_code="69990", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now)
    line_surg_prefix = ProcessableClaimLineItem(id=3, claim_db_id=1, line_number=3, service_date=date.today(), procedure_code="SURGXYZ", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now)
    line_norm = ProcessableClaimLineItem(id=4, claim_db_id=1, line_number=4, service_date=date.today(), procedure_code="99213", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now)
    line_nonsurg_prefix = ProcessableClaimLineItem(id=5, claim_db_id=1, line_number=5, service_date=date.today(), procedure_code="A1234", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now)

    assert feature_extractor._detect_surgery_codes([line_surg_cpt_range_start, line_norm]) == 1.0
    assert feature_extractor._detect_surgery_codes([line_surg_cpt_range_end, line_norm]) == 1.0
    assert feature_extractor._detect_surgery_codes([line_surg_prefix, line_norm]) == 1.0
    assert feature_extractor._detect_surgery_codes([line_norm, line_nonsurg_prefix]) == 0.0
    assert feature_extractor._detect_surgery_codes([]) == 0.0
    assert feature_extractor._detect_surgery_codes([ProcessableClaimLineItem(id=6, claim_db_id=1, line_number=1, service_date=date.today(), procedure_code=None, units=1, charge_amount=Decimal(10), created_at=now,updated_at=now)]) == 0.0


def test_calculate_complexity_score_detailed(feature_extractor: FeatureExtractor):
    now = datetime.now(timezone.utc)
    lines_low = [ProcessableClaimLineItem(id=1, claim_db_id=1, line_number=1, service_date=date.today(), procedure_code="P1", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now)] # 1 line, 1 unit
    assert feature_extractor._calculate_complexity_score(lines_low) == pytest.approx((1/10.0) + (1/20.0)) # 0.1 + 0.05 = 0.15

    lines_medium = [
        ProcessableClaimLineItem(id=1, claim_db_id=1, line_number=1, service_date=date.today(), procedure_code="P1", units=3, charge_amount=Decimal(10), created_at=now,updated_at=now),
        ProcessableClaimLineItem(id=2, claim_db_id=1, line_number=2, service_date=date.today(), procedure_code="P2", units=4, charge_amount=Decimal(10), created_at=now,updated_at=now)
    ] # 2 lines, 7 units
    assert feature_extractor._calculate_complexity_score(lines_medium) == pytest.approx((2/10.0) + (7/20.0)) # 0.2 + 0.35 = 0.55

    lines_high_lines = [ProcessableClaimLineItem(id=i, claim_db_id=1, line_number=i, service_date=date.today(), procedure_code=f"P{i}", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now) for i in range(12)] # 12 lines, 12 units
    assert feature_extractor._calculate_complexity_score(lines_high_lines) == pytest.approx(0.5 + (12/20.0)) # 0.5 (capped lines) + 0.6 = 1.0 (capped at 1.0)

    lines_high_units = [ProcessableClaimLineItem(id=1, claim_db_id=1, line_number=1, service_date=date.today(), procedure_code="P1", units=25, charge_amount=Decimal(10), created_at=now,updated_at=now)] # 1 line, 25 units
    assert feature_extractor._calculate_complexity_score(lines_high_units) == pytest.approx((1/10.0) + 0.5) # 0.1 + 0.5 (capped units) = 0.6

    lines_max_all = [ProcessableClaimLineItem(id=i, claim_db_id=1, line_number=i, service_date=date.today(), procedure_code=f"P{i}", units=10, charge_amount=Decimal(10), created_at=now,updated_at=now) for i in range(10)] # 10 lines, 100 units
    assert feature_extractor._calculate_complexity_score(lines_max_all) == 1.0 # Capped score

    assert feature_extractor._calculate_complexity_score([]) == 0.0

# --- Test for main extract_features method ---
def test_extract_features_valid_claim(feature_extractor: FeatureExtractor):
    # This test replaces the old test_extract_features_returns_correct_shape_and_type
    expected_feature_count = 7 # FeatureExtractor now produces a fixed set of 7 features

    test_dob = date(1980, 1, 15)
    test_service_from = date(2023, 1, 1)
    test_service_to = date(2023, 1, 3) # Duration 3 days
    test_total_charges = Decimal("250.75")
    test_num_lines = 2
    test_insurance = "medicare"

    claim = create_test_claim(
        dob=test_dob,
        service_from=test_service_from,
        service_to=test_service_to,
        total_charges=test_total_charges,
        num_lines=test_num_lines,
        insurance_type=test_insurance
    )
    features = feature_extractor.extract_features(claim)

    assert isinstance(features, np.ndarray), "Features should be a NumPy array"
    assert features.shape == (1, expected_feature_count), \
        f"Expected shape (1, {expected_feature_count}), but got {features.shape}"
    assert features.dtype == np.float32, "Features should be float32 type"

    assert np.isclose(features[0, 0], np.log1p(float(test_total_charges)))
    assert np.isclose(features[0, 1], float(test_num_lines))
    expected_age = (test_service_from - test_dob).days / 365.25
    assert np.isclose(features[0, 2], expected_age)
    expected_duration = float((test_service_to - test_service_from).days + 1)
    assert np.isclose(features[0, 3], expected_duration)
    assert np.isclose(features[0, 4], INSURANCE_TYPE_MAPPING.get(test_insurance.lower()))
    # _detect_surgery_codes for default P1, P2 codes will be 0.0
    assert np.isclose(features[0, 5], 0.0)
    # _calculate_complexity_score for 2 lines, 1 unit each: (2/10) + (2/20) = 0.2 + 0.1 = 0.3
    expected_complexity = (2.0/10.0) + (2.0*1.0/20.0) # num_lines/10 + total_units/20
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
    assert feature_extractor.extract_features(claim) is None # Expect None on error

def test_feature_extractor_initialization(feature_extractor: FeatureExtractor):
    assert isinstance(feature_extractor, FeatureExtractor)
    # If FeatureExtractor's __init__ loaded settings for feature_count, test it here.
    # Current version does not store feature_count from settings in __init__.
    # It produces a fixed number of features (7).
    # settings = get_settings()
    # assert feature_extractor.feature_count == settings.ML_FEATURE_COUNT # If it were stored
