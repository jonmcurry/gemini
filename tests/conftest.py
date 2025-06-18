import pytest
import os
import asyncio # Added import
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator, Generator

from claims_processor.src.core.database.db_session import Base, get_db_session # App's Base and session getter
from claims_processor.src.main import app # Your FastAPI app
# from claims_processor.src.core.config.settings import get_settings # Not directly used here, TEST_DATABASE_URL is read from env

# Use a separate test database URL
TEST_DB_URL_DEFAULT = "postgresql+asyncpg://postgres:postgres@localhost:5432/claims_processor_test_db"
# It's better to get this from environment variable for flexibility
# The settings object could also provide this, but direct env var is also common for test setups.
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", TEST_DB_URL_DEFAULT)


# Create a new engine instance for testing
# Use connect_args for things like statement_cache_size if needed, e.g., {"statement_cache_size": 0} for some tests.
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False) # echo=False for less verbose test output

# Create a new session local for testing
AsyncTestSessionLocal = sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

async def override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Override for FastAPI's dependency_overrides."""
    async with AsyncTestSessionLocal() as session:
        yield session
        # No explicit rollback here, as tests might want to commit to check DB state.
        # Overall cleanup is handled by setup_test_database fixture.

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]: # Type hint for event_loop
    # Make event loop session-scoped for async fixtures
    # policy = asyncio.get_event_loop_policy() # Not needed for Python 3.8+ default policy
    loop = asyncio.new_event_loop() # Simpler way to get a new loop
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="session", autouse=True)
async def setup_test_database(event_loop): # event_loop fixture is used implicitly by async fixtures
    """
    This fixture runs once per session.
    It creates all tables and then drops them after all tests in the session are done.
    """
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield # Tests run here

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await test_engine.dispose() # Clean up the engine connections

@pytest.fixture()
async def db_session(setup_test_database) -> AsyncGenerator[AsyncSession, None]:
    """
    This fixture provides a clean session for each test function.
    It uses the AsyncTestSessionLocal.
    """
    async with AsyncTestSessionLocal() as session:
        yield session
        # If tests perform commits and you want to ensure each test starts clean
        # *without* relying solely on drop_all/create_all for inter-test isolation,
        # you might start a transaction here and roll it back.
        # await session.rollback()
        # However, for this setup, individual tests can manage their transactions if needed,
        # or the full DB wipe (drop_all/create_all) handles broader isolation.

@pytest.fixture()
def client(db_session: AsyncSession) -> Generator[any, None, None]: # Using 'any' for TestClient as it's not defined here yet
    """
    Provides a FastAPI TestClient that uses the overridden database session.
    """
    # Override the app's dependency for get_db_session to use the test session
    app.dependency_overrides[get_db_session] = lambda: db_session # Use the specific db_session instance for this test

    from fastapi.testclient import TestClient # Import here to avoid issues if fastapi isn't installed when conftest is first parsed

    with TestClient(app) as c:
        yield c

    # Clean up overrides after test to prevent leakage between tests
    app.dependency_overrides.clear()
