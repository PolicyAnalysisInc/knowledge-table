from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, create_model

from app.models.llm_responses import BaseResponseModel
from app.services.llm.pydantic_llm_service import PydanticCompletionService
from app.core.config import Settings # Import Settings for fixture


# Fixture for PydanticCompletionService
@pytest.fixture
def pydantic_service(test_settings, mocker):
    # Patch the Agent class within the service module
    mock_agent_instance = AsyncMock() # This is the instance the patched Agent will return
    mock_agent_class = mocker.patch(
        "app.services.llm.pydantic_llm_service.Agent",
        return_value=mock_agent_instance
    )

    # Mock get_active_llm_config *within the fixture* to return a default config.
    # Individual tests can override this patch if needed.
    mock_default_llm_instance = MagicMock()
    mock_default_config = MagicMock()
    mock_default_config.get_instance.return_value = mock_default_llm_instance
    mock_default_config.settings = {"default_setting": True}
    mock_get_active = mocker.patch(
        "app.services.llm.pydantic_llm_service.get_active_llm_config",
        return_value=mock_default_config
    )

    # Initialize the service - it will now get the mocked Agent instance
    service = PydanticCompletionService(test_settings)

    # Return the service and the mocks for assertions in tests
    return service, mock_agent_instance, mock_agent_class, mock_get_active


@pytest.mark.asyncio
async def test_generate_completion(pydantic_service):
    service, mock_agent_instance, _, mock_get_active = pydantic_service
    # No skip check needed as Agent is always patched
    class DummyResponseModel(BaseModel):
        content: str

    # Mock the response from agent.run to include the .output attribute
    mock_response_instance = DummyResponseModel(content="Test response")
    mock_agent_run_result = MagicMock()
    mock_agent_run_result.output = mock_response_instance
    mock_agent_instance.run = AsyncMock(return_value=mock_agent_run_result)

    result = await service.generate_completion("Test prompt", DummyResponseModel)

    assert isinstance(result, DummyResponseModel)
    assert result.content == "Test response"
    mock_get_active.assert_called_once() # Ensure default config was used
    mock_agent_instance.run.assert_awaited_once_with("Test prompt", **{"default_setting": True})


@pytest.mark.asyncio
async def test_generate_completion_with_override(pydantic_service, mocker):
    service, mock_agent_instance, mock_agent_class, mock_get_active = pydantic_service
    # No skip check needed

    # Inherit from BaseResponseModel now
    class DummyResponseModel(BaseResponseModel):
        content: str

    # Create an override LLMConfig
    mock_override_llm_instance = MagicMock()
    override_config = MagicMock()
    override_config.get_instance.return_value = mock_override_llm_instance
    override_config.settings = {"override_setting": 0.99}
    # Mock provider and model_name needed for agent name construction
    override_config.provider = "mock_provider"
    override_config.model_name = "mock_model"

    # Mock the response from agent.run
    mock_response_instance = DummyResponseModel(content="Override response", confidence=10, reasoning="Test")
    mock_agent_run_result = MagicMock()
    mock_agent_run_result.output = mock_response_instance
    mock_agent_instance.run = AsyncMock(return_value=mock_agent_run_result)

    result = await service.generate_completion(
        "Test prompt override",
        DummyResponseModel,
        llm_config_override=override_config
    )

    assert isinstance(result, DummyResponseModel)
    assert result.content == "Override response"

    # Assertions
    mock_get_active.assert_not_called() # Should NOT use the default getter
    override_config.get_instance.assert_called_once() # Should use the override's instance getter
    
    # Capture the arguments passed to the Agent constructor
    mock_agent_class.assert_called_once()
    call_args, call_kwargs = mock_agent_class.call_args
    
    # Check the LLM instance argument
    assert call_args[0] == mock_override_llm_instance
    
    # Check keyword arguments
    assert call_kwargs.get("instrument") is True
    expected_agent_name = f"Agent-{override_config.provider}-{override_config.model_name}"
    assert call_kwargs.get("name") == expected_agent_name
    
    # Check the output_type argument more specifically
    actual_output_type = call_kwargs.get("output_type")
    assert actual_output_type is not None
    assert issubclass(actual_output_type, BaseModel) # Should still be a Pydantic model
    assert hasattr(actual_output_type, 'model_fields')
    # Verify it has the fields from DummyResponseModel *except* all_responses
    assert 'content' in actual_output_type.model_fields
    assert 'confidence' in actual_output_type.model_fields # Inherited from BaseResponseModel
    assert 'reasoning' in actual_output_type.model_fields # Inherited from BaseResponseModel
    assert 'all_responses' not in actual_output_type.model_fields # Should be excluded
    
    mock_agent_instance.run.assert_awaited_once_with( # Run should use override settings
        "Test prompt override", **{"override_setting": 0.99}
    )


@pytest.mark.asyncio
async def test_generate_completion_none_response(pydantic_service):
    service, mock_agent_instance, _, mock_get_active = pydantic_service
    # No skip check needed
    class DummyResponseModel(BaseModel):
        content: str

    # Mock agent.run to return a result object with .output = None
    mock_agent_run_result = MagicMock()
    mock_agent_run_result.output = None
    mock_agent_instance.run = AsyncMock(return_value=mock_agent_run_result)

    result = await service.generate_completion("Test prompt", DummyResponseModel)

    assert result is None
    mock_get_active.assert_called_once() # Ensure default config was used
    mock_agent_instance.run.assert_awaited_once_with("Test prompt", **{"default_setting": True})

@pytest.mark.asyncio
async def test_generate_completion_all_fields_none(pydantic_service):
    service, mock_agent_instance, _, mock_get_active = pydantic_service
    # No skip check needed
    class DummyResponseModelOptional(BaseModel):
        content: str | None = None
        another_field: int | None = None

    # Mock agent.run to return result object with .output = instance with None fields
    mock_response_instance = DummyResponseModelOptional(content=None, another_field=None)
    mock_agent_run_result = MagicMock()
    mock_agent_run_result.output = mock_response_instance
    mock_agent_instance.run = AsyncMock(return_value=mock_agent_run_result)

    result = await service.generate_completion("Test prompt", DummyResponseModelOptional)

    # The service should return None if all fields are None
    assert result is None
    mock_get_active.assert_called_once() # Ensure default config was used
    mock_agent_instance.run.assert_awaited_once_with("Test prompt", **{"default_setting": True})


@pytest.mark.asyncio
async def test_generate_completion_agent_not_initialized(test_settings, mocker):
    # Create service without API key
    settings_no_key = test_settings.model_copy(update={"openai_api_key": None, "gemini_api_key": None})
    service = PydanticCompletionService(settings_no_key)

    # Mock the get_instance call within the scope of this test 
    # to simulate failure due to missing keys
    mock_active_config = MagicMock()
    mock_active_config.get_instance.side_effect = RuntimeError("Simulated instantiation failure")
    # Keep the patch specific to this test
    mock_get_active = mocker.patch(
        "app.services.llm.pydantic_llm_service.get_active_llm_config",
        return_value=mock_active_config
    )

    class DummyResponseModel(BaseModel):
        content: str

    # Expect None because the mocked get_instance raises an error, 
    # which should be caught by the try/except block in generate_completion
    result = await service.generate_completion("Test prompt", DummyResponseModel)
    assert result is None
    mock_get_active.assert_called_once()


@pytest.mark.asyncio
async def test_decompose_query(pydantic_service):
    service, mock_agent_instance, _, mock_get_active = pydantic_service
    # No skip check needed
    test_query = "Test query"
    # Currently, decompose_query returns the original query as a placeholder
    # We don't mock agent.run because the current implementation doesn't call it
    result = await service.decompose_query(test_query)

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