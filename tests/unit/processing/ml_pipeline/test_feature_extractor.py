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
    NON_NUMERIC_SURGERY_INDICATORS
)
from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
# Remove direct get_settings import if not used elsewhere, as FeatureExtractor internally calls it.
# from claims_processor.src.core.config.settings import get_settings
from claims_processor.src.core.config.settings import Settings # For mock_settings_for_feature_cache
from claims_processor.src.core.monitoring.app_metrics import MetricsCollector # For mock_metrics_collector
from unittest.mock import MagicMock, patch, call # For mocking
import time # For TTL tests
import hashlib # For cache key generation checks

# --- Fixtures for Caching Tests ---

@pytest.fixture
def mock_metrics_collector():
    mc = MagicMock(spec=MetricsCollector)
    mc.record_cache_operation = MagicMock()
    return mc

@pytest.fixture
def mock_settings_for_feature_cache():
    return Settings(
        ML_FEATURE_CACHE_MAXSIZE=10,
        ML_FEATURE_CACHE_TTL=0.1,    # Short TTL for testing
        DATABASE_URL="sqlite+aiosqlite:///:memory:",
        APP_ENCRYPTION_KEY="test_key_must_be_32_bytes_long"
    )

@pytest.fixture
def feature_extractor(mock_metrics_collector, mock_settings_for_feature_cache):
    # Patch get_settings specifically where it's imported in feature_extractor.py
    with patch('claims_processor.src.processing.ml_pipeline.feature_extractor.get_settings', return_value=mock_settings_for_feature_cache):
        extractor = FeatureExtractor(metrics_collector=mock_metrics_collector)
    return extractor

# Helper to create ProcessableClaim instances for cache tests
def create_test_processable_claim(claim_id="C001", total_charges=Decimal("100.00"), line_count=1, dob_year=1990, service_days_offset=0, insurance_type="commercial ppo") -> ProcessableClaim:
    now = datetime.now(timezone.utc)
    line_items_list = [
        ProcessableClaimLineItem(
            id=i+1, claim_db_id=1, line_number=i+1,
            service_date=date(2024, 1, 15 + i + service_days_offset),
            procedure_code=f"P{i:03d}", units=1,
            charge_amount=(total_charges / Decimal(str(line_count))) if line_count > 0 else Decimal("0.0"),
            rvu_total=None, created_at=now, updated_at=now
        ) for i in range(line_count)
    ] if line_count > 0 else []

    return ProcessableClaim(
        id=1, claim_id=claim_id, facility_id="F001", patient_account_number=f"ACC_{claim_id}", # Ensure unique acc num
        total_charges=total_charges,
        service_from_date=date(2024, 1, 10 + service_days_offset),
        service_to_date=date(2024, 1, 10 + service_days_offset + (line_count if line_count > 0 else 1)),
        patient_date_of_birth=date(dob_year, 1, 1),
        line_items=line_items_list,
        medical_record_number=f"MRN_{claim_id}", insurance_type=insurance_type, financial_class="CASH", # Use passed insurance_type
        processing_status="pending",
        created_at=now,
        updated_at=now,
        validation_errors=None, rvu_adjustments_details=None, ml_model_version_used=None,
        ml_score=None, ml_derived_decision=None, processing_duration_ms=None
    )

# List of actual feature calculation methods to patch/spy on for cache tests
INTERNAL_CALCULATION_METHODS_TO_PATCH = [
    "_normalize_total_charges",
    "_calculate_patient_age",
    "_calculate_service_duration",
    "_encode_insurance_type",
    "_detect_surgery_codes",
    "_calculate_complexity_score"
]


# --- Existing Tests (modified fixture name if necessary) ---
# The existing tests used a simple `feature_extractor()` fixture.
# To avoid conflicts and keep them working if they don't need caching specifics,
# we can rename their fixture or adapt them. For now, assume they might need updating
# if FeatureExtractor() without args is no longer valid.
# Let's define a simple extractor for them if they don't interact with cache/metrics.
@pytest.fixture
def simple_feature_extractor() -> FeatureExtractor:
    # This is a workaround if existing tests break due to new __init__ requiring metrics.
    # Ideally, all tests should use the fixture that provides necessary mocks.
    # For now, assuming FeatureExtractor() might not be directly possible if get_settings() call in __init__
    # needs specific settings not provided by default when no mock is active.
    # The new `feature_extractor` fixture above is for caching tests.
    # The tests below this fixture were written assuming `FeatureExtractor()` works without args.
    # Let's assume that the constructor `FeatureExtractor(metrics_collector=MagicMock())` is a valid
    # way to get a simple extractor if the new `__init__` is `__init__(self, metrics_collector: MetricsCollector)`.
    # And that it internally calls get_settings() which will pick up some default Settings().
    # This part might need adjustment based on how strictly Settings are needed by the raw constructor.
    # For safety, let's assume the old tests need to be adapted or use a specifically mocked extractor.
    # Given the previous change, FeatureExtractor now REQUIRES metrics_collector.
    # So, the old tests need a metrics_collector.
    dummy_settings = Settings(ML_FEATURE_CACHE_MAXSIZE=0, ML_FEATURE_CACHE_TTL=0) # Cache effectively off
    with patch('claims_processor.src.processing.ml_pipeline.feature_extractor.get_settings', return_value=dummy_settings):
        return FeatureExtractor(metrics_collector=MagicMock(spec=MetricsCollector))


# Updated create_test_claim helper
def create_test_claim(
    dob: Optional[date] = date(1990, 1, 1),
    service_from: date = date(2023, 1, 1),
    service_to: date = date(2023, 1, 3),
    total_charges: Decimal = Decimal("100.00"),
    num_lines: int = 2,
    line_item_details: Optional[List[Dict[str, Any]]] = None,
    insurance_type: Optional[str] = "commercial",
    financial_class: Optional[str] = "FC1"
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
# These tests should now use simple_feature_extractor to avoid issues with caching logic / metrics
def test_calculate_patient_age_detailed(simple_feature_extractor: FeatureExtractor): # Use simple_feature_extractor
    dob = date(1990, 6, 15)
    assert simple_feature_extractor._calculate_patient_age(dob, date(2023, 6, 14)) == 32.0
    assert simple_feature_extractor._calculate_patient_age(dob, date(2023, 6, 15)) == 33.0
    assert simple_feature_extractor._calculate_patient_age(dob, date(2023, 6, 16)) == 33.0
    assert simple_feature_extractor._calculate_patient_age(None, date.today()) is None
    assert simple_feature_extractor._calculate_patient_age("not-a-date", date.today()) is None # type: ignore

def test_calculate_service_duration_detailed(simple_feature_extractor: FeatureExtractor): # Use simple_feature_extractor
    claim_same_day = create_test_claim(service_from=date(2023, 1, 1), service_to=date(2023, 1, 1))
    assert simple_feature_extractor._calculate_service_duration(claim_same_day) == 1.0
    claim_multi_day = create_test_claim(service_from=date(2023, 1, 1), service_to=date(2023, 1, 3))
    assert simple_feature_extractor._calculate_service_duration(claim_multi_day) == 3.0
    claim_invalid_dates = create_test_claim(service_from=date(2023, 1, 3), service_to=date(2023, 1, 1))
    assert simple_feature_extractor._calculate_service_duration(claim_invalid_dates) == 0.0
    claim_no_from_date = create_test_claim(service_from=None, service_to=date(2023,1,1)) # type: ignore
    assert simple_feature_extractor._calculate_service_duration(claim_no_from_date) == 0.0
    claim_no_to_date = create_test_claim(service_from=date(2023,1,1), service_to=None) # type: ignore
    assert simple_feature_extractor._calculate_service_duration(claim_no_to_date) == 0.0


def test_normalize_total_charges_detailed(simple_feature_extractor: FeatureExtractor): # Use simple_feature_extractor
    assert simple_feature_extractor._normalize_total_charges(Decimal("0.0")) == np.log1p(0.0)
    assert simple_feature_extractor._normalize_total_charges(Decimal("100.0")) == np.log1p(100.0)
    assert simple_feature_extractor._normalize_total_charges(Decimal("99999.0")) == np.log1p(99999.0)
    # Test with float input, though Pydantic model uses Decimal
    assert simple_feature_extractor._normalize_total_charges(100.0) == np.log1p(100.0) # type: ignore
    # The method expects Decimal from Pydantic, but test with float for robustness of float() cast
    with pytest.raises(TypeError): # Or handle gracefully if that's the design
         simple_feature_extractor._normalize_total_charges("invalid_type") # type: ignore


def test_encode_insurance_type_detailed(simple_feature_extractor: FeatureExtractor): # Use simple_feature_extractor
    assert simple_feature_extractor._encode_insurance_type("Medicare") == INSURANCE_TYPE_MAPPING.get("medicare")
    assert simple_feature_extractor._encode_insurance_type(" medicaid ") == INSURANCE_TYPE_MAPPING.get("medicaid") # Test stripping
    assert simple_feature_extractor._encode_insurance_type("Commercial") == INSURANCE_TYPE_MAPPING.get("commercial")
    assert simple_feature_extractor._encode_insurance_type("SELF-PAY") == INSURANCE_TYPE_MAPPING.get("self-pay")
    # Check if "hmo" and "ppo" are in INSURANCE_TYPE_MAPPING or use DEFAULT_INSURANCE_ENCODING
    assert simple_feature_extractor._encode_insurance_type("HMO") == INSURANCE_TYPE_MAPPING.get("hmo", DEFAULT_INSURANCE_ENCODING)
    assert simple_feature_extractor._encode_insurance_type("PPO") == INSURANCE_TYPE_MAPPING.get("ppo", DEFAULT_INSURANCE_ENCODING)
    assert simple_feature_extractor._encode_insurance_type("other") == INSURANCE_TYPE_MAPPING.get("other")
    assert simple_feature_extractor._encode_insurance_type("UnknownType") == DEFAULT_INSURANCE_ENCODING
    assert simple_feature_extractor._encode_insurance_type(None) == DEFAULT_INSURANCE_ENCODING

def test_detect_surgery_codes_detailed(simple_feature_extractor: FeatureExtractor): # Use simple_feature_extractor
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
    assert simple_feature_extractor._calculate_complexity_score(claim_15l_30u_nosurg, 0.0) == pytest.approx(0.4 + 0.3)

    # With surgery (flag = 1.0, adds WEIGHT_SURGERY = 0.3 to score)
    claim_1l_1u_surg = create_test_claim(num_lines=1, line_item_details=[{"units": 1}])
    # Expected: 0.04 (lines) + 0.015 (units) + 0.3 (surgery) = 0.355
    assert simple_feature_extractor._calculate_complexity_score(claim_1l_1u_surg, 1.0) == pytest.approx(0.04 + 0.015 + 0.3)

    claim_10l_20u_surg = create_test_claim(num_lines=10, line_item_details=[{"units": 2}]*10) # 10 lines, 20 units
    # Expected: 0.4 (lines) + 0.3 (units) + 0.3 (surgery) = 1.0
    assert simple_feature_extractor._calculate_complexity_score(claim_10l_20u_surg, 1.0) == pytest.approx(1.0)

    claim_empty_lines_nosurg = create_test_claim(num_lines=0, line_item_details=[])
    assert simple_feature_extractor._calculate_complexity_score(claim_empty_lines_nosurg, 0.0) == 0.0
    claim_empty_lines_surg = create_test_claim(num_lines=0, line_item_details=[])
    # The logic for complexity score uses the surgery_detected_flag passed in. If lines are empty,
    # _detect_surgery_codes would return 0.0. If complexity is called with 1.0 externally, it would add 0.3.
    # The current _calculate_complexity_score takes the flag as an argument.
    assert simple_feature_extractor._calculate_complexity_score(claim_empty_lines_surg, 1.0) == pytest.approx(0.3) # Only surgery weight


# --- Test for main extract_features method ---
# These tests should also use simple_feature_extractor if they don't test caching.
def test_extract_features_valid_claim(simple_feature_extractor: FeatureExtractor): # Use simple_feature_extractor
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


def test_extract_features_with_surgery(simple_feature_extractor: FeatureExtractor): # Use simple_feature_extractor
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


def test_extract_features_missing_dob(simple_feature_extractor: FeatureExtractor): # Use simple_feature_extractor
    claim = create_test_claim(dob=None)
    features = simple_feature_extractor.extract_features(claim)
    assert features is not None
    assert features.shape == (7,) # Check 1D shape
    assert np.isclose(features[2], -1.0) # Imputed age

def test_extract_features_handles_error_in_helper(simple_feature_extractor: FeatureExtractor, monkeypatch): # Use simple_feature_extractor
    claim = create_test_claim()
    def mock_raise_error(*args, **kwargs):
        raise ValueError("Test error in helper")
    monkeypatch.setattr(simple_feature_extractor, '_calculate_service_duration', mock_raise_error)
    assert simple_feature_extractor.extract_features(claim) is None # Expect None on error

def test_feature_extractor_initialization(simple_feature_extractor: FeatureExtractor): # Use simple_feature_extractor
    assert isinstance(simple_feature_extractor, FeatureExtractor)
    # This test now implicitly checks that FeatureExtractor can be initialized with a mock MetricsCollector.

# --- Tests for Caching Logic ---

def test_feature_cache_key_stability_for_model_dump_json():
    """ Test that model_dump_json is stable for cache key generation. """
    claim1_v1 = create_test_processable_claim(claim_id="C001", total_charges=Decimal("100.00"))
    claim1_v2 = create_test_processable_claim(claim_id="C001", total_charges=Decimal("100.00")) # Identical
    claim2_diff_id = create_test_processable_claim(claim_id="C002", total_charges=Decimal("100.00"))
    claim3_diff_charge = create_test_processable_claim(claim_id="C001", total_charges=Decimal("200.00"))

    # Fields to exclude as per implementation
    exclude_fields = {'processing_status', 'validation_errors', 'rvu_adjustments_details', 'ml_model_version_used'}

    json_v1 = claim1_v1.model_dump_json(exclude=exclude_fields)
    json_v2 = claim1_v2.model_dump_json(exclude=exclude_fields)
    json_c2 = claim2_diff_id.model_dump_json(exclude=exclude_fields)
    json_c3 = claim3_diff_charge.model_dump_json(exclude=exclude_fields)

    assert json_v1 == json_v2, "JSON dumps of identical claims should be identical for cache key."
    assert hashlib.sha256(json_v1.encode('utf-8')).hexdigest() == hashlib.sha256(json_v2.encode('utf-8')).hexdigest()

    assert json_v1 != json_c2, "JSON dumps of different claims (ID) should differ."
    assert hashlib.sha256(json_v1.encode('utf-8')).hexdigest() != hashlib.sha256(json_c2.encode('utf-8')).hexdigest()

    assert json_v1 != json_c3, "JSON dumps of different claims (charge) should differ."
    assert hashlib.sha256(json_v1.encode('utf-8')).hexdigest() != hashlib.sha256(json_c3.encode('utf-8')).hexdigest()


def test_feature_cache_hit(feature_extractor: FeatureExtractor, mock_metrics_collector: MagicMock):
    claim = create_test_processable_claim(claim_id="CHIT01")

    # Manually generate expected cache key
    claim_json_for_key = claim.model_dump_json(exclude={'processing_status', 'validation_errors', 'rvu_adjustments_details', 'ml_model_version_used'})
    cache_key = hashlib.sha256(claim_json_for_key.encode('utf-8')).hexdigest()

    mock_features = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    feature_extractor.feature_cache[cache_key] = mock_features

    # Patch all internal calculation methods on the instance to ensure they are not called
    for method_name in INTERNAL_CALCULATION_METHODS_TO_PATCH:
        setattr(feature_extractor, method_name, MagicMock(side_effect=RuntimeError(f"{method_name} should not be called on cache hit")))

    extracted_features = feature_extractor.extract_features(claim)

    assert np.array_equal(extracted_features, mock_features)
    mock_metrics_collector.record_cache_operation.assert_called_once_with(
        cache_type='ml_feature_cache', operation_type='get', outcome='hit'
    )
    # Assert that none of the patched calculation methods were called
    for method_name in INTERNAL_CALCULATION_METHODS_TO_PATCH:
        getattr(feature_extractor, method_name).assert_not_called()

def test_feature_cache_miss_then_hit(feature_extractor: FeatureExtractor, mock_metrics_collector: MagicMock):
    claim = create_test_processable_claim(claim_id="CMISS01")

    # Spy on internal methods by patching them on the instance
    original_methods = {}
    mocked_methods_on_instance = {}
    for method_name in INTERNAL_CALCULATION_METHODS_TO_PATCH:
        original_methods[method_name] = getattr(feature_extractor, method_name)
        # Create a mock that calls the original method (spy behavior)
        mocked_method = MagicMock(wraps=original_methods[method_name])
        setattr(feature_extractor, method_name, mocked_method)
        mocked_methods_on_instance[method_name] = mocked_method

    # First call - Cache Miss
    features1 = feature_extractor.extract_features(claim)
    assert features1 is not None
    mock_metrics_collector.record_cache_operation.assert_any_call(
        cache_type='ml_feature_cache', operation_type='get', outcome='miss'
    )
    for method_name in INTERNAL_CALCULATION_METHODS_TO_PATCH:
        mocked_methods_on_instance[method_name].assert_called()
        mocked_methods_on_instance[method_name].reset_mock() # Reset for the next call check

    # Manually check if in cache (optional, as next call implies it)
    claim_json_for_key = claim.model_dump_json(exclude={'processing_status', 'validation_errors', 'rvu_adjustments_details', 'ml_model_version_used'})
    cache_key = hashlib.sha256(claim_json_for_key.encode('utf-8')).hexdigest()
    assert cache_key in feature_extractor.feature_cache
    assert np.array_equal(feature_extractor.feature_cache[cache_key], features1)

    # Second call - Cache Hit
    features2 = feature_extractor.extract_features(claim)
    assert np.array_equal(features1, features2)
    mock_metrics_collector.record_cache_operation.assert_called_with( # Last call should be hit
        cache_type='ml_feature_cache', operation_type='get', outcome='hit'
    )
    for method_name in INTERNAL_CALCULATION_METHODS_TO_PATCH:
        mocked_methods_on_instance[method_name].assert_not_called() # Should not be called on hit

    # Restore original methods (cleanup, good practice)
    for method_name, original_method in original_methods.items():
        setattr(feature_extractor, method_name, original_method)


def test_feature_cache_ttl_expiration(feature_extractor: FeatureExtractor, mock_metrics_collector: MagicMock, mock_settings_for_feature_cache: Settings):
    claim = create_test_processable_claim(claim_id="CTTL01")

    # Spy on calculation methods as in miss_then_hit
    original_methods = {}
    mocked_methods_on_instance = {}
    for method_name in INTERNAL_CALCULATION_METHODS_TO_PATCH:
        original_methods[method_name] = getattr(feature_extractor, method_name)
        mocked_method = MagicMock(wraps=original_methods[method_name])
        setattr(feature_extractor, method_name, mocked_method)
        mocked_methods_on_instance[method_name] = mocked_method

    initial_time = time.time()
    with patch('time.time') as mock_time:
        # First call - Cache Miss
        mock_time.return_value = initial_time
        features1 = feature_extractor.extract_features(claim)
        assert features1 is not None
        mock_metrics_collector.record_cache_operation.assert_any_call(
            cache_type='ml_feature_cache', operation_type='get', outcome='miss'
        )
        for method_name in INTERNAL_CALCULATION_METHODS_TO_PATCH:
            mocked_methods_on_instance[method_name].assert_called()
            mocked_methods_on_instance[method_name].reset_mock()

        # Second call - Still within TTL (cache hit)
        mock_time.return_value = initial_time + mock_settings_for_feature_cache.ML_FEATURE_CACHE_TTL * 0.5
        features2 = feature_extractor.extract_features(claim)
        assert np.array_equal(features1, features2)
        mock_metrics_collector.record_cache_operation.assert_called_with( # Last call is hit
            cache_type='ml_feature_cache', operation_type='get', outcome='hit'
        )
        for method_name in INTERNAL_CALCULATION_METHODS_TO_PATCH:
            mocked_methods_on_instance[method_name].assert_not_called()

        # Third call - After TTL expiration (cache miss again)
        mock_time.return_value = initial_time + mock_settings_for_feature_cache.ML_FEATURE_CACHE_TTL + 0.1 # Advance time past TTL
        features3 = feature_extractor.extract_features(claim)
        assert features3 is not None # Should recalculate
        # Check that it's a miss again
        # The record_cache_operation mock stores all calls, check the sequence or the last relevant one
        # For simplicity, we check if 'miss' was called as part of the new set of calls
        calls = [
            call(cache_type='ml_feature_cache', operation_type='get', outcome='miss'), # Initial call
            call(cache_type='ml_feature_cache', operation_type='get', outcome='hit'),   # Second call (hit)
            call(cache_type='ml_feature_cache', operation_type='get', outcome='miss')  # Third call (miss after TTL)
        ]
        mock_metrics_collector.record_cache_operation.assert_has_calls(calls, any_order=False) # Check sequence

        for method_name in INTERNAL_CALCULATION_METHODS_TO_PATCH:
            mocked_methods_on_instance[method_name].assert_called() # Should be called again

    # Restore original methods
    for method_name, original_method in original_methods.items():
        setattr(feature_extractor, method_name, original_method)


def test_feature_cache_different_claims_different_keys(feature_extractor: FeatureExtractor, mock_metrics_collector: MagicMock):
    claim1 = create_test_processable_claim(claim_id="CDIFF01", total_charges=Decimal("100"))
    claim2 = create_test_processable_claim(claim_id="CDIFF02", total_charges=Decimal("200")) # Different ID and charges

    # Spy on calculation methods for claim1
    # (Not strictly necessary to spy for this test, but good for confirming no unexpected hits)

    # Call for claim1
    features1 = feature_extractor.extract_features(claim1)
    assert features1 is not None
    calls_claim1 = [call(cache_type='ml_feature_cache', operation_type='get', outcome='miss')]
    mock_metrics_collector.record_cache_operation.assert_has_calls(calls_claim1)

    # Call for claim2
    features2 = feature_extractor.extract_features(claim2)
    assert features2 is not None
    calls_claim2 = calls_claim1 + [call(cache_type='ml_feature_cache', operation_type='get', outcome='miss')]
    mock_metrics_collector.record_cache_operation.assert_has_calls(calls_claim2, any_order=False) # Both are misses

    assert not np.array_equal(features1, features2) # Features should be different
    assert len(feature_extractor.feature_cache) == 2 # Both should be cached separately

    # Verify cache keys are different
    claim1_json = claim1.model_dump_json(exclude={'processing_status', 'validation_errors', 'rvu_adjustments_details', 'ml_model_version_used'})
    key1 = hashlib.sha256(claim1_json.encode('utf-8')).hexdigest()
    claim2_json = claim2.model_dump_json(exclude={'processing_status', 'validation_errors', 'rvu_adjustments_details', 'ml_model_version_used'})
    key2 = hashlib.sha256(claim2_json.encode('utf-8')).hexdigest()

    assert key1 != key2
    assert key1 in feature_extractor.feature_cache
    assert key2 in feature_extractor.feature_cache
