import pytest
from datetime import date
from claims_processor.scripts.database.manage_claims_production_partitions import (
    get_partition_name as get_cp_partition_name,
    generate_create_partition_ddl as generate_cp_ddl,
    PARENT_TABLE_NAME as CP_PARENT_TABLE_NAME
)

def test_get_cp_partition_name():
    assert get_cp_partition_name(2025) == "claims_production_y2025"
    assert get_cp_partition_name(2030) == "claims_production_y2030"

def test_generate_cp_ddl():
    year = 2026
    start_date = date(year, 1, 1)
    end_date = date(year + 1, 1, 1) # Start of next year
    partition_name = get_cp_partition_name(year)

    expected_ddl = (
        f"CREATE TABLE IF NOT EXISTS {partition_name} "
        f"PARTITION OF {CP_PARENT_TABLE_NAME} "
        f"FOR VALUES FROM ('{start_date.strftime('%Y-%m-%d')}') "
        f"TO ('{end_date.strftime('%Y-%m-%d')}')"
    )
    assert generate_cp_ddl(year) == expected_ddl
