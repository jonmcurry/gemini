import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Callable, Tuple
import hashlib # Added

from sqlalchemy.ext.asyncio import AsyncSession

from claims_processor.src.core.security.audit_logger_service import AuditLoggerService
from claims_processor.src.core.database.models.audit_log_db import AuditLogModel

@pytest.fixture
def mock_db_session_factory() -> Tuple[MagicMock, AsyncMock]: # Returns factory_mock, session_mock
    mock_session = AsyncMock(spec=AsyncSession)
    # Mock the begin_nested or begin method if used by the service, or just add for general session behavior
    # AuditLoggerService uses session.begin() as an async context manager
    mock_session_begin_cm = AsyncMock() # This is the context manager returned by session.begin()
    mock_session_begin_cm.__aenter__.return_value = mock_session # Simulate entering 'async with session.begin()'
    mock_session_begin_cm.__aexit__.return_value = None # Simulate successful exit from 'async with session.begin()'
    mock_session.begin = MagicMock(return_value=mock_session_begin_cm) # session.begin() returns the CM

    mock_session.add = MagicMock()
    # mock_session.commit = AsyncMock() # Not needed as session.begin() handles commit
    # mock_session.rollback = AsyncMock() # Not needed as session.begin() handles rollback

    # The AuditLoggerService expects a factory that, when called, returns an async context manager
    # which in turn yields the session.
    async_cm_factory_yields = AsyncMock() # This is the context manager returned by db_session_factory()
    async_cm_factory_yields.__aenter__.return_value = mock_session # The session object
    async_cm_factory_yields.__aexit__.return_value = None # Simulate successful exit from 'async with db_session_factory()'

    mock_session_factory_instance = MagicMock(spec=Callable[[], Any]) # Mock the factory callable itself
    mock_session_factory_instance.return_value = async_cm_factory_yields # Factory call returns the context manager

    return mock_session_factory_instance, mock_session


@pytest.fixture
def audit_logger_service(mock_db_session_factory: Tuple[MagicMock, AsyncMock]) -> AuditLoggerService:
    factory_mock, _ = mock_db_session_factory
    return AuditLoggerService(db_session_factory=factory_mock)

@pytest.mark.asyncio
async def test_log_event_success(audit_logger_service: AuditLoggerService, mock_db_session_factory: Tuple[MagicMock, AsyncMock]):
    factory_mock, mock_session = mock_db_session_factory

    success_flag = await audit_logger_service.log_event( # Renamed 'success' to 'success_flag' to avoid keyword clash
        action="TEST_ACTION",
        success=True,
        resource="TestResource",
        resource_id="res_123",
        user_id="test_user",
        patient_id_to_hash="patient_abc",
        details={"key": "value"},
        client_ip="127.0.0.1",
        user_agent="TestAgent/1.0",
        session_id="sess_xyz"
    )

    assert success_flag is True
    mock_session.add.assert_called_once()
    added_log_entry = mock_session.add.call_args[0][0]

    assert isinstance(added_log_entry, AuditLogModel)
    assert added_log_entry.action == "TEST_ACTION"
    assert added_log_entry.success is True
    assert added_log_entry.resource == "TestResource"
    assert added_log_entry.resource_id == "res_123"
    assert added_log_entry.user_id == "test_user"

    test_patient_id_val = "patient_abc" # The value passed in the call
    expected_hash = hashlib.sha256(test_patient_id_val.encode('utf-8')).hexdigest()
    assert added_log_entry.patient_id_hash == expected_hash

    assert added_log_entry.details == {"key": "value"}
    assert added_log_entry.ip_address == "127.0.0.1"
    assert added_log_entry.user_agent == "TestAgent/1.0"
    assert added_log_entry.session_id == "sess_xyz"
    assert added_log_entry.failure_reason is None

    # Check that the db_session_factory was called to get the session context manager
    factory_mock.assert_called_once()
    # Check that the session context manager was entered
    factory_mock.return_value.__aenter__.assert_called_once()
    # Check that session.begin() context manager was entered
    mock_session.begin.assert_called_once()
    mock_session.begin.return_value.__aenter__.assert_called_once()


@pytest.mark.asyncio
async def test_log_event_failure_db_error_on_add(audit_logger_service: AuditLoggerService, mock_db_session_factory: Tuple[MagicMock, AsyncMock]):
    factory_mock, mock_session = mock_db_session_factory

    mock_session.add.side_effect = Exception("Database add error")

    success_flag = await audit_logger_service.log_event(action="DB_FAIL_ACTION", success=True)

    assert success_flag is False
    mock_session.add.assert_called_once()
    # session.begin().__aexit__ should have been called with an exception type if 'add' fails within 'session.begin()' block
    # This means the transaction would attempt to rollback.
    mock_session.begin.return_value.__aexit__.assert_called_once()
    assert mock_session.begin.return_value.__aexit__.call_args[0][0] is not None # exc_type should be set


@pytest.mark.asyncio
async def test_log_event_failure_db_error_on_commit(audit_logger_service: AuditLoggerService, mock_db_session_factory: Tuple[MagicMock, AsyncMock]):
    factory_mock, mock_session = mock_db_session_factory

    # Simulate error when session.begin() context manager exits (tries to commit)
    mock_session.begin.return_value.__aexit__.side_effect = Exception("Database commit error")

    success_flag = await audit_logger_service.log_event(action="DB_FAIL_COMMIT_ACTION", success=True)

    assert success_flag is False
    mock_session.add.assert_called_once() # Add was successful
    mock_session.begin.return_value.__aexit__.assert_called_once() # __aexit__ was called and raised error
    # When side_effect is an exception, the call_args might not be what we expect for normal exit.
    # The important part is that __aexit__ was called and an exception occurred, leading to success_flag = False.


@pytest.mark.asyncio
async def test_log_event_failure_reason_populated(audit_logger_service: AuditLoggerService, mock_db_session_factory: Tuple[MagicMock, AsyncMock]):
    factory_mock, mock_session = mock_db_session_factory

    await audit_logger_service.log_event(
        action="FAILED_ACTION",
        success=False,
        failure_reason="Something went wrong"
    )

    mock_session.add.assert_called_once()
    added_log_entry = mock_session.add.call_args[0][0]
    assert added_log_entry.success is False
    assert added_log_entry.failure_reason == "Something went wrong"

@pytest.mark.asyncio
async def test_log_event_default_user_id(audit_logger_service: AuditLoggerService, mock_db_session_factory: Tuple[MagicMock, AsyncMock]):
    factory_mock, mock_session = mock_db_session_factory
    await audit_logger_service.log_event(action="SYSTEM_ACTION", success=True)

    mock_session.add.assert_called_once()
    added_log_entry = mock_session.add.call_args[0][0]
    assert added_log_entry.user_id == "system"

# Removed test_log_event_patient_id_hashing_long_id as SHA256 is fixed length output, no truncation.

@pytest.mark.asyncio
async def test_log_event_patient_id_hashing_none(audit_logger_service: AuditLoggerService, mock_db_session_factory: Tuple[MagicMock, AsyncMock]):
    factory_mock, mock_session = mock_db_session_factory

    await audit_logger_service.log_event(action="HASH_TEST_NONE", success=True, patient_id_to_hash=None)

    mock_session.add.assert_called_once()
    added_log_entry = mock_session.add.call_args[0][0]
    assert added_log_entry.patient_id_hash is None
```
