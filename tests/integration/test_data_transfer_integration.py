import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func # Added func for count
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from typing import Optional # Added Optional

from claims_processor.src.core.database.models.claims_db import ClaimModel, ClaimLineItemModel
from claims_processor.src.core.database.models.claims_production_db import ClaimsProductionModel
from claims_processor.src.api.routes.data_transfer_routes import run_data_transfer_background

# Test data helper
def create_staging_claim_for_transfer(
    claim_id_suffix: str,
    ml_score_val: Optional[Decimal],
    duration_ms_val: Optional[int],
    days_ago: int = 0,
    service_date_override: Optional[date] = None # For specific partition testing
) -> ClaimModel:
    now = datetime.now(timezone.utc)
    created_time = now - timedelta(days=days_ago)

    # Use provided service_date_override or calculate one based on days_ago
    service_date_val = service_date_override if service_date_override else (date(2023, 7, 15) - timedelta(days=days_ago))


    return ClaimModel(
        claim_id=f"TRANSFER_C_{claim_id_suffix}",
        facility_id="F_TRANSFER",
        patient_account_number=f"P_TRANSFER_{claim_id_suffix}",
        service_from_date=service_date_val, # Use the determined service_date_val
        service_to_date=service_date_val + timedelta(days=1),
        total_charges=Decimal("500.00"),
        processing_status="processing_complete",
        transferred_to_prod_at=None,
        created_at=created_time,
        updated_at=created_time,
        processed_at=created_time,
        ml_score=ml_score_val,
        ml_derived_decision="ML_APPROVED" if ml_score_val and ml_score_val >= Decimal("0.8") else ("ML_REJECTED" if ml_score_val is not None else "N/A"),
        processing_duration_ms=duration_ms_val,
        line_items=[
            ClaimLineItemModel(line_number=1, service_date=service_date_val, procedure_code="P001", units=1, charge_amount=Decimal("500.00"))
        ]
    )

@pytest.mark.asyncio
async def test_data_transfer_successful_e2e(db_session: AsyncSession):
    # 1. Seed Data
    # Ensure service_from_date aligns with created partitions (e.g., 2023, 2024, 2025)
    claim_to_transfer_1 = create_staging_claim_for_transfer("T1", Decimal("0.9"), 120, days_ago=2, service_date_override=date(2023,1,15))
    claim_to_transfer_2 = create_staging_claim_for_transfer("T2", Decimal("0.6"), 150, days_ago=1, service_date_override=date(2024,1,15))

    claim_pending = ClaimModel(claim_id="PENDING_C", facility_id="F_PEND", patient_account_number="P_PEND",
                               service_from_date=date(2023,2,1), service_to_date=date(2023,2,2),
                               total_charges=Decimal(100), processing_status="pending",
                               created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))

    already_transferred_claim_data = {
        "claim_id": "ALREADY_T", "facility_id": "F_ALR", "patient_account_number": "P_ALR",
        "service_from_date": date(2023,2,5), "service_to_date": date(2023,2,6), "total_charges": Decimal(200),
        "processing_status": "processing_complete",
        "transferred_to_prod_at": datetime.now(timezone.utc) - timedelta(days=5),
        "created_at":datetime.now(timezone.utc), "updated_at":datetime.now(timezone.utc)
    }
    claim_already_transferred = ClaimModel(**already_transferred_claim_data)

    db_session.add_all([claim_to_transfer_1, claim_to_transfer_2, claim_pending, claim_already_transferred])
    await db_session.commit()

    transfer_id_1 = claim_to_transfer_1.id
    transfer_id_2 = claim_to_transfer_2.id

    # 2. Execute the data transfer logic
    transfer_summary = await run_data_transfer_background(limit=10, client_ip=None, user_agent_header=None)

    assert transfer_summary["successfully_transferred"] == 2
    assert transfer_summary["selected_from_staging"] == 2

    # 3. Verify data in claims_production table
    prod_claim_1 = await db_session.get(ClaimsProductionModel, (transfer_id_1, claim_to_transfer_1.service_from_date))
    assert prod_claim_1 is not None
    assert prod_claim_1.claim_id == claim_to_transfer_1.claim_id
    assert prod_claim_1.ml_prediction_score == claim_to_transfer_1.ml_score
    assert prod_claim_1.processing_duration_ms == claim_to_transfer_1.processing_duration_ms
    assert prod_claim_1.risk_category == "LOW"
    assert prod_claim_1.created_at_prod is not None

    prod_claim_2 = await db_session.get(ClaimsProductionModel, (transfer_id_2, claim_to_transfer_2.service_from_date))
    assert prod_claim_2 is not None
    assert prod_claim_2.claim_id == claim_to_transfer_2.claim_id
    assert prod_claim_2.risk_category == "MEDIUM"

    # 4. Verify staging claims are updated
    await db_session.refresh(claim_to_transfer_1)
    await db_session.refresh(claim_to_transfer_2)
    assert claim_to_transfer_1.transferred_to_prod_at is not None
    assert claim_to_transfer_2.transferred_to_prod_at is not None

    # 5. Verify other staging claims are untouched
    await db_session.refresh(claim_pending)
    assert claim_pending.transferred_to_prod_at is None
    assert claim_pending.processing_status == "pending"

    await db_session.refresh(claim_already_transferred)
    assert claim_already_transferred.transferred_to_prod_at is not None

@pytest.mark.asyncio
async def test_data_transfer_no_eligible_claims(db_session: AsyncSession):
    now = datetime.now(timezone.utc)
    claim_pending = ClaimModel(claim_id="PENDING_ONLY", facility_id="F_PEND", patient_account_number="P_PEND",
                               service_from_date=date(2023,3,1), service_to_date=date(2023,3,2),
                               total_charges=Decimal(100), processing_status="pending",
                               created_at=now, updated_at=now)
    already_transferred_claim_data = {
        "claim_id": "ALREADY_T_ONLY", "facility_id": "F_ALR", "patient_account_number": "P_ALR",
        "service_from_date": date(2023,3,5), "service_to_date": date(2023,3,6), "total_charges": Decimal(200),
        "processing_status": "processing_complete",
        "transferred_to_prod_at": now - timedelta(days=5),
        "created_at":now, "updated_at":now
    }
    claim_already_transferred = ClaimModel(**already_transferred_claim_data)
    db_session.add_all([claim_pending, claim_already_transferred])
    await db_session.commit()

    transfer_summary = await run_data_transfer_background(limit=10, client_ip=None, user_agent_header=None)

    assert transfer_summary["message"] == "No claims to transfer."
    assert transfer_summary["successfully_transferred"] == 0
    assert transfer_summary["selected_from_staging"] == 0

    result_prod_count = await db_session.execute(select(func.count(ClaimsProductionModel.id))) # Corrected count
    assert result_prod_count.scalar_one() == 0
