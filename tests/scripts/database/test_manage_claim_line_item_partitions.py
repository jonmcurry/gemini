import pytest
from datetime import date
from claims_processor.scripts.database.manage_claim_line_item_partitions import (
    get_partition_name as get_cli_partition_name,
    generate_create_partition_ddl as generate_cli_ddl,
    PARENT_TABLE_NAME as CLI_PARENT_TABLE_NAME
)

def test_get_cli_partition_name():
    assert get_cli_partition_name(2024, 5) == "claim_line_items_y2024m05"
    assert get_cli_partition_name(2025, 12) == "claim_line_items_y2025m12"

def test_generate_cli_ddl():
    year, month = 2024, 7
    start_date = date(year, month, 1)
    end_date = date(year, 8, 1) # Next month
    partition_name = get_cli_partition_name(year, month)

    expected_ddl = (
        f"CREATE TABLE IF NOT EXISTS {partition_name} "
        f"PARTITION OF {CLI_PARENT_TABLE_NAME} "
        f"FOR VALUES FROM ('{start_date.strftime('%Y-%m-%d')}') "
        f"TO ('{end_date.strftime('%Y-%m-%d')}')"
    )
    assert generate_cli_ddl(year, month) == expected_ddl

def test_generate_cli_ddl_december():
    year, month = 2024, 12
    start_date = date(year, month, 1)
    end_date = date(year + 1, 1, 1) # Next year, Jan
    partition_name = get_cli_partition_name(year, month)

    expected_ddl = (
        f"CREATE TABLE IF NOT EXISTS {partition_name} "
        f"PARTITION OF {CLI_PARENT_TABLE_NAME} "
        f"FOR VALUES FROM ('{start_date.strftime('%Y-%m-%d')}') "
        f"TO ('{end_date.strftime('%Y-%m-%d')}')"
    )
    assert generate_cli_ddl(year, month) == expected_ddl
