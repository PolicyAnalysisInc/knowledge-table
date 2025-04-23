from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from app.services.llm.pydantic_llm_service import PydanticCompletionService
from app.core.config import Settings # Import Settings for fixture


# Fixture for PydanticCompletionService
@pytest.fixture
def pydantic_service(test_settings, mocker):
    # Patch the Agent class within the service module
    mock_agent_instance = AsyncMock() # This is the instance the patched Agent will return
    mocker.patch(
        "app.services.llm.pydantic_llm_service.Agent",
        return_value=mock_agent_instance
    )

    # Initialize the service - it will now get the mocked Agent instance
    service = PydanticCompletionService(test_settings)

    # We can optionally return the mock instance too if tests need to access the class mock itself,
    # but here the tests interact with service.agent, which is the mock_agent_instance.
    # For clarity, let's ensure service.agent is indeed our mock instance
    service.agent = mock_agent_instance # Explicitly assign for clarity, though patch should handle it

    return service


@pytest.mark.asyncio
async def test_generate_completion(pydantic_service):
    # No skip check needed as Agent is always patched
    class DummyResponseModel(BaseModel):
        content: str

    # Mock the response from agent.run to include the .output attribute
    mock_response_instance = DummyResponseModel(content="Test response")
    mock_agent_run_result = MagicMock()
    mock_agent_run_result.output = mock_response_instance
    pydantic_service.agent.run = AsyncMock(return_value=mock_agent_run_result)

    result = await pydantic_service.generate_completion("Test prompt", DummyResponseModel)

    assert isinstance(result, DummyResponseModel)
    assert result.content == "Test response"
    pydantic_service.agent.run.assert_awaited_once_with("Test prompt", **{})


@pytest.mark.asyncio
async def test_generate_completion_none_response(pydantic_service):
    # No skip check needed
    class DummyResponseModel(BaseModel):
        content: str

    # Mock agent.run to return a result object with .output = None
    mock_agent_run_result = MagicMock()
    mock_agent_run_result.output = None
    pydantic_service.agent.run = AsyncMock(return_value=mock_agent_run_result)

    result = await pydantic_service.generate_completion("Test prompt", DummyResponseModel)

    assert result is None
    pydantic_service.agent.run.assert_awaited_once_with("Test prompt", **{})

@pytest.mark.asyncio
async def test_generate_completion_all_fields_none(pydantic_service):
    # No skip check needed
    class DummyResponseModelOptional(BaseModel):
        content: str | None = None
        another_field: int | None = None

    # Mock agent.run to return result object with .output = instance with None fields
    mock_response_instance = DummyResponseModelOptional(content=None, another_field=None)
    mock_agent_run_result = MagicMock()
    mock_agent_run_result.output = mock_response_instance
    pydantic_service.agent.run = AsyncMock(return_value=mock_agent_run_result)

    result = await pydantic_service.generate_completion("Test prompt", DummyResponseModelOptional)

    # The service should return None if all fields are None
    assert result is None
    pydantic_service.agent.run.assert_awaited_once_with("Test prompt", **{})


@pytest.mark.asyncio
async def test_generate_completion_agent_not_initialized(test_settings, mocker):
    # Create service without API key
    settings_no_key = test_settings.model_copy(update={"openai_api_key": None, "gemini_api_key": None})
    service = PydanticCompletionService(settings_no_key)

    # Mock the get_instance call within the scope of this test 
    # to simulate failure due to missing keys
    mock_active_config = MagicMock()
    mock_active_config.get_instance.side_effect = RuntimeError("Simulated instantiation failure")
    mocker.patch(
        "app.services.llm.pydantic_llm_service.get_active_llm_config",
        return_value=mock_active_config
    )

    class DummyResponseModel(BaseModel):
        content: str

    # Expect None because the mocked get_instance raises an error, 
    # which should be caught by the try/except block in generate_completion
    result = await service.generate_completion("Test prompt", DummyResponseModel)
    assert result is None


@pytest.mark.asyncio
async def test_decompose_query(pydantic_service):
    # No skip check needed
    test_query = "Test query"
    # Currently, decompose_query returns the original query as a placeholder
    # We don't mock agent.run because the current implementation doesn't call it
    result = await pydantic_service.decompose_query(test_query)

    assert result == {"sub_queries": [test_query]}
    # If decompose_query were implemented using agent.run, add assertion like:
    # pydantic_service.agent.run.assert_awaited_once_with(...)


@pytest.mark.asyncio
async def test_decompose_query_agent_not_initialized(test_settings):
    # Create service without API key to ensure agent instantiation might fail
    settings_no_key = test_settings.model_copy(update={"openai_api_key": None, "gemini_api_key": None}) # Ensure relevant keys are None
    service = PydanticCompletionService(settings_no_key)

    # Since decompose_query is not implemented, it just returns the placeholder
    # If it were implemented, we'd expect it to return None or raise error
    query = "Decompose this"
    expected_result = {"sub_queries": [query]}
    result = await service.decompose_query(query)
    assert result == expected_result # Current behavior 