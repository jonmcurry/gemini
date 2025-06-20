import pytest
import numpy as np
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Optional, Any

from claims_processor.src.processing.ml_pipeline.feature_extractor import (
    FeatureExtractor,
    INSURANCE_TYPE_MAPPING,
    DEFAULT_INSURANCE_ENCODING,
    SURGERY_CPT_RANGES, # Updated
    NON_NUMERIC_SURGERY_INDICATORS # Updated
)
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
    assert feature_extractor._encode_insurance_type("HMO") == INSURANCE_TYPE_MAPPING.get("hmo")
    assert feature_extractor._encode_insurance_type("PPO") == INSURANCE_TYPE_MAPPING.get("ppo")
    assert feature_extractor._encode_insurance_type("other") == INSURANCE_TYPE_MAPPING.get("other")
    assert feature_extractor._encode_insurance_type("UnknownType") == DEFAULT_INSURANCE_ENCODING
    assert feature_extractor._encode_insurance_type(None) == DEFAULT_INSURANCE_ENCODING

def test_detect_surgery_codes_detailed(feature_extractor: FeatureExtractor):
    now = datetime.now(timezone.utc)
    # CPT surgery codes (inclusive range 10021-69990)
    line_surg_cpt_exact_start = ProcessableClaimLineItem(id=1,line_number=1, claim_db_id=1,service_date=date.today(), procedure_code="10021", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now)
    line_surg_cpt_exact_end = ProcessableClaimLineItem(id=2,line_number=1, claim_db_id=1,service_date=date.today(), procedure_code="69990", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now)
    line_surg_cpt_middle = ProcessableClaimLineItem(id=3,line_number=1, claim_db_id=1,service_date=date.today(), procedure_code="35000", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now)
    line_surg_cpt_with_alpha = ProcessableClaimLineItem(id=4,line_number=1, claim_db_id=1,service_date=date.today(), procedure_code="12345A", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now) # Assuming numeric part is checked

    # Non-surgery CPT codes
    line_norm_cpt = ProcessableClaimLineItem(id=5,line_number=1, claim_db_id=1,service_date=date.today(), procedure_code="99213", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now) # E/M code
    line_cpt_just_outside_low = ProcessableClaimLineItem(id=6,line_number=1, claim_db_id=1,service_date=date.today(), procedure_code="10020", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now)
    line_cpt_just_outside_high = ProcessableClaimLineItem(id=7,line_number=1, claim_db_id=1,service_date=date.today(), procedure_code="70000", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now)

    # HCPCS Level II codes
    line_surg_hcpcs_s = ProcessableClaimLineItem(id=8,line_number=1, claim_db_id=1,service_date=date.today(), procedure_code="S2065", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now) # Example 'S' code
    line_nonsurg_hcpcs_a = ProcessableClaimLineItem(id=9,line_number=1, claim_db_id=1,service_date=date.today(), procedure_code="A1234", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now)

    assert feature_extractor._detect_surgery_codes([line_surg_cpt_exact_start]) == 1.0
    assert feature_extractor._detect_surgery_codes([line_surg_cpt_exact_end]) == 1.0
    assert feature_extractor._detect_surgery_codes([line_surg_cpt_middle]) == 1.0
    assert feature_extractor._detect_surgery_codes([line_surg_cpt_with_alpha]) == 1.0 # Assuming 12345 is in range

    assert feature_extractor._detect_surgery_codes([line_norm_cpt]) == 0.0
    assert feature_extractor._detect_surgery_codes([line_cpt_just_outside_low]) == 0.0
    assert feature_extractor._detect_surgery_codes([line_cpt_just_outside_high]) == 0.0

    assert feature_extractor._detect_surgery_codes([line_surg_hcpcs_s]) == 1.0
    assert feature_extractor._detect_surgery_codes([line_nonsurg_hcpcs_a]) == 0.0

    assert feature_extractor._detect_surgery_codes([line_surg_cpt_middle, line_norm_cpt]) == 1.0 # If any line is surgery
    assert feature_extractor._detect_surgery_codes([line_norm_cpt, line_nonsurg_hcpcs_a]) == 0.0
    assert feature_extractor._detect_surgery_codes([]) == 0.0
    assert feature_extractor._detect_surgery_codes([ProcessableClaimLineItem(id=10,line_number=1, claim_db_id=1,service_date=date.today(), procedure_code=None, units=1, charge_amount=Decimal(10), created_at=now,updated_at=now)]) == 0.0
    assert feature_extractor._detect_surgery_codes([ProcessableClaimLineItem(id=11,line_number=1, claim_db_id=1,service_date=date.today(), procedure_code="INVALID", units=1, charge_amount=Decimal(10), created_at=now,updated_at=now)]) == 0.0


def test_calculate_complexity_score_detailed(feature_extractor: FeatureExtractor):
    # Test cases: (num_lines, total_units, surgery_flag) -> expected_score
    # score_from_lines = min(num_lines / MAX_LINES_FOR_SCORE_CONTRIB, 1.0) * WEIGHT_LINES
    # score_from_units = min(total_units / MAX_UNITS_FOR_SCORE_CONTRIB, 1.0) * WEIGHT_UNITS
    # score_from_surgery = surgery_detected_flag * WEIGHT_SURGERY

    # No surgery (flag = 0.0)
    claim_1l_1u_nosurg = create_test_claim(num_lines=1, line_item_details=[{"units": 1}]) # 1 line, 1 unit
    # Expected: lines = min(1/10,1)*0.4 = 0.1*0.4=0.04. units = min(1/20,1)*0.3 = 0.05*0.3=0.015. surgery=0. Total=0.055
    assert feature_extractor._calculate_complexity_score(claim_1l_1u_nosurg, 0.0) == pytest.approx(0.04 + 0.015)

    claim_5l_10u_nosurg = create_test_claim(num_lines=5, line_item_details=[{"units": 2}]*5) # 5 lines, 10 units
    # Expected: lines = min(5/10,1)*0.4 = 0.5*0.4=0.2. units = min(10/20,1)*0.3 = 0.5*0.3=0.15. surgery=0. Total=0.35
    assert feature_extractor._calculate_complexity_score(claim_5l_10u_nosurg, 0.0) == pytest.approx(0.2 + 0.15)

    claim_10l_20u_nosurg = create_test_claim(num_lines=10, line_item_details=[{"units": 2}]*10) # 10 lines, 20 units
    # Expected: lines = min(10/10,1)*0.4 = 1*0.4=0.4. units = min(20/20,1)*0.3 = 1*0.3=0.3. surgery=0. Total=0.7
    assert feature_extractor._calculate_complexity_score(claim_10l_20u_nosurg, 0.0) == pytest.approx(0.4 + 0.3)

    claim_15l_30u_nosurg = create_test_claim(num_lines=15, line_item_details=[{"units": 2}]*15) # 15 lines, 30 units
    # Expected: lines = min(15/10,1)*0.4 = 1*0.4=0.4. units = min(30/20,1)*0.3 = 1*0.3=0.3. surgery=0. Total=0.7
    assert feature_extractor._calculate_complexity_score(claim_15l_30u_nosurg, 0.0) == pytest.approx(0.4 + 0.3)

    # With surgery (flag = 1.0, adds WEIGHT_SURGERY = 0.3 to score)
    claim_1l_1u_surg = create_test_claim(num_lines=1, line_item_details=[{"units": 1}])
    # Expected: 0.04 (lines) + 0.015 (units) + 0.3 (surgery) = 0.355
    assert feature_extractor._calculate_complexity_score(claim_1l_1u_surg, 1.0) == pytest.approx(0.04 + 0.015 + 0.3)

    claim_10l_20u_surg = create_test_claim(num_lines=10, line_item_details=[{"units": 2}]*10) # 10 lines, 20 units
    # Expected: 0.4 (lines) + 0.3 (units) + 0.3 (surgery) = 1.0
    assert feature_extractor._calculate_complexity_score(claim_10l_20u_surg, 1.0) == pytest.approx(1.0)

    claim_empty_lines_nosurg = create_test_claim(num_lines=0, line_item_details=[])
    assert feature_extractor._calculate_complexity_score(claim_empty_lines_nosurg, 0.0) == 0.0
    claim_empty_lines_surg = create_test_claim(num_lines=0, line_item_details=[])
    assert feature_extractor._calculate_complexity_score(claim_empty_lines_surg, 1.0) == 0.0 # No lines, so surgery on what? Score should be 0. Or 0.3 if surgery flag means claim-level surgery. The current logic is line-based.

# --- Test for main extract_features method ---
def test_extract_features_valid_claim(feature_extractor: FeatureExtractor):
    expected_feature_count = 7

    test_dob = date(1980, 1, 15)
    test_service_from = date(2023, 1, 1)
    test_service_to = date(2023, 1, 3)
    test_total_charges = Decimal("250.75")
    test_num_lines = 2
    test_insurance = "medicare"
    # Line items that do NOT indicate surgery by default from create_test_claim
    claim_no_surgery = create_test_claim(
        dob=test_dob, service_from=test_service_from, service_to=test_service_to,
        total_charges=test_total_charges, num_lines=test_num_lines, insurance_type=test_insurance
    )
    features_no_surgery = feature_extractor.extract_features(claim_no_surgery)

    assert isinstance(features_no_surgery, np.ndarray), "Features should be a NumPy array"
    assert features_no_surgery.shape == (expected_feature_count,), \
        f"Expected shape ({expected_feature_count},), but got {features_no_surgery.shape}"
    assert features_no_surgery.dtype == np.float32, "Features should be float32 type"

    assert np.isclose(features_no_surgery[0], np.log1p(float(test_total_charges)))
    assert np.isclose(features_no_surgery[1], float(test_num_lines))
    expected_age = (test_service_from - test_dob).days / 365.25 # Approx age
    assert np.isclose(features_no_surgery[2], expected_age)
    expected_duration = float((test_service_to - test_service_from).days + 1)
    assert np.isclose(features_no_surgery[3], expected_duration)
    assert np.isclose(features_no_surgery[4], INSURANCE_TYPE_MAPPING.get(test_insurance.lower(), DEFAULT_INSURANCE_ENCODING))

    surgery_flag_no_surgery = 0.0 # Based on default P1, P2 codes in create_test_claim
    assert np.isclose(features_no_surgery[5], surgery_flag_no_surgery)

    # Complexity for claim_no_surgery: 2 lines, 2 units total (1 per line default), no surgery
    # lines_score = min(2/10,1)*0.4 = 0.2*0.4=0.08
    # units_score = min(2/20,1)*0.3 = 0.1*0.3=0.03
    # surgery_score = 0.0
    # total = 0.08 + 0.03 = 0.11
    expected_complexity_no_surgery = 0.08 + 0.03
    assert np.isclose(features_no_surgery[6], expected_complexity_no_surgery)


def test_extract_features_with_surgery(feature_extractor: FeatureExtractor):
    expected_feature_count = 7
    claim_with_surgery = create_test_claim(
        num_lines=3, # This will be overridden by line_item_details length
        line_item_details=[
            {"procedure_code": "99213", "units": 1},
            {"procedure_code": "12345", "units": 2}, # Surgery (CPT range)
            {"procedure_code": "S2065", "units": 1}  # Surgery (HCPCS S-code)
        ]
    ) # 3 lines (P1, 12345, S2065), total units 1+2+1=4. Surgery detected.
    features_surgery = feature_extractor.extract_features(claim_with_surgery)
    assert features_surgery is not None
    assert features_surgery.shape == (expected_feature_count,)

    surgery_flag_with_surgery = 1.0
    assert np.isclose(features_surgery[5], surgery_flag_with_surgery)

    # Complexity for claim_with_surgery: 3 lines, 4 units, surgery detected
    # lines_score = min(3/10,1)*0.4 = 0.3*0.4 = 0.12
    # units_score = min(4/20,1)*0.3 = 0.2*0.3 = 0.06
    # surgery_score = 1.0 * 0.3 = 0.3
    # total = 0.12 + 0.06 + 0.3 = 0.48
    expected_complexity_with_surgery = 0.48
    assert np.isclose(features_surgery[6], expected_complexity_with_surgery)


def test_extract_features_missing_dob(feature_extractor: FeatureExtractor):
    claim = create_test_claim(dob=None)
    features = feature_extractor.extract_features(claim)
    assert features is not None
    assert features.shape == (7,) # Check 1D shape
    assert np.isclose(features[2], -1.0) # Imputed age

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
