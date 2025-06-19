import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession # For type hinting the mock session

from claims_processor.src.core.security.audit_logger_service import AuditLoggerService
from claims_processor.src.core.database.models.audit_log_db import AuditLogModel

@pytest.fixture
def mock_db_session() -> MagicMock:
    session = MagicMock(spec=AsyncSession)
    session.add = MagicMock()
    # Configure commit and refresh to be awaitable (async)
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session

@pytest.fixture
def audit_logger_service(mock_db_session: MagicMock) -> AuditLoggerService:
    return AuditLoggerService(db_session=mock_db_session)

@pytest.mark.asyncio
async def test_log_event_success(
    audit_logger_service: AuditLoggerService,
    mock_db_session: MagicMock
):
    await audit_logger_service.log_event(
        action="TEST_ACTION_SUCCESS",
        success=True,
        user_id="test_user",
        resource="TestResource",
        resource_id="123",
        ip_address="127.0.0.1",
        user_agent="TestAgent/1.0",
        details={"key": "value"}
    )

    mock_db_session.add.assert_called_once()
    added_object = mock_db_session.add.call_args[0][0]

    assert isinstance(added_object, AuditLogModel)
    assert added_object.action == "TEST_ACTION_SUCCESS"
    assert added_object.success is True
    assert added_object.user_id == "test_user"
    assert added_object.resource == "TestResource"
    assert added_object.resource_id == "123"
    assert added_object.ip_address == "127.0.0.1"
    assert added_object.user_agent == "TestAgent/1.0"
    assert added_object.details == {"key": "value"}
    assert added_object.failure_reason is None

    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once_with(added_object)

@pytest.mark.asyncio
async def test_log_event_failure(
    audit_logger_service: AuditLoggerService,
    mock_db_session: MagicMock
):
    failure_msg = "Something went wrong"
    await audit_logger_service.log_event(
        action="TEST_ACTION_FAILURE",
        success=False,
        user_id="test_user_fail",
        failure_reason=failure_msg,
        details={"error_code": 500}
    )

    mock_db_session.add.assert_called_once()
    added_object = mock_db_session.add.call_args[0][0]

    assert isinstance(added_object, AuditLogModel)
    assert added_object.action == "TEST_ACTION_FAILURE"
    assert added_object.success is False
    assert added_object.user_id == "test_user_fail"
    assert added_object.failure_reason == failure_msg
    assert added_object.details == {"error_code": 500}

    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once_with(added_object)

@pytest.mark.asyncio
@patch('claims_processor.src.core.security.audit_logger_service.logger') # Patch the logger
async def test_log_event_db_commit_exception(
    mock_logger: MagicMock, # Injected by @patch
    audit_logger_service: AuditLoggerService,
    mock_db_session: MagicMock
):
    mock_db_session.commit.side_effect = Exception("DB commit failed")

    await audit_logger_service.log_event(
        action="TEST_DB_FAIL",
        success=True
    )

    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_not_called()

    mock_logger.error.assert_called_once()
    args, kwargs = mock_logger.error.call_args
    assert "Failed to save audit log to database" in args[0]
    assert kwargs.get('action') == "TEST_DB_FAIL"
    assert "DB commit failed" in kwargs.get('original_error', '')

@pytest.mark.asyncio
async def test_log_event_failure_reason_only_if_not_success(
    audit_logger_service: AuditLoggerService,
    mock_db_session: MagicMock
):
    await audit_logger_service.log_event(
        action="TEST_SUCCESS_WITH_REASON",
        success=True,
        failure_reason="This should be ignored"
    )

    added_object_success = mock_db_session.add.call_args[0][0]
    assert added_object_success.failure_reason is None

    mock_db_session.add.reset_mock() # Reset for the next call assertion

    reason = "Actual failure"
    await audit_logger_service.log_event(
        action="TEST_FAILURE_WITH_REASON",
        success=False,
        failure_reason=reason
    )
    added_object_failure = mock_db_session.add.call_args[0][0]
    assert added_object_failure.failure_reason == reason
