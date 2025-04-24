"""Tests for the PydanticMultiCompletionService."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.models.llm_responses import BaseResponseModel, StrResponseModel, IntResponseModel
from app.services.llm.llm_configuration import LLMConfig
from app.services.llm.pydantic_llm_service import (
    PydanticMultiCompletionService,
    PydanticCompletionService
)

# Dummy Response Model for testing
class DummyResponse(StrResponseModel):
    pass # Inherits confidence, reasoning, all_responses from Base

# --- Fixtures --- #

@pytest.fixture
def mock_settings() -> Settings:
    """Returns a mock Settings object."""
    return Settings() # Adjust if settings require specific mock values

@pytest.fixture
def mock_llm_configs() -> dict[str, LLMConfig]:
    """Provides a dictionary of mock LLMConfig objects for testing."""
    return {
        "key1": LLMConfig(model_name="model1", provider="prov1", settings={"temp": 0.1}),
        "key2": LLMConfig(model_name="model2", provider="prov2", settings={"temp": 0.2}),
        "key3": LLMConfig(model_name="model3", provider="prov3", settings={"temp": 0.3}),
        # Add default keys used in the service
        "gemini-2.5": LLMConfig(model_name="gemini-2.5-pro-preview-03-25", provider="google-gla", settings={}),
        "gemini-2.5-flash": LLMConfig(model_name="gemini-2.5-flash-preview-04-17", provider="google-gla", settings={}),
        "gpt-4o": LLMConfig(model_name="gpt-4o", provider="openai", settings={}),
        "gpt-4.1-mini": LLMConfig(model_name="gpt-4.1-mini-2025-04-14", provider="openai", settings={}),
    }

@pytest.fixture
def multi_service(mock_settings) -> PydanticMultiCompletionService:
    """Provides an instance of PydanticMultiCompletionService."""
    # No need to mock parent __init__ separately, just instantiate the child
    return PydanticMultiCompletionService(settings=mock_settings)

# --- Test Cases --- #

@pytest.mark.asyncio
async def test_multi_completion_first_success(
    multi_service: PydanticMultiCompletionService,
    mock_llm_configs: dict[str, LLMConfig],
    mocker: MagicMock
):
    """Test successful completion where the first LLM call succeeds."""
    mock_parent_generate = mocker.patch(
        "app.services.llm.pydantic_llm_service.PydanticCompletionService.generate_completion",
        new_callable=AsyncMock
    )

    # Define results for each call
    result1 = DummyResponse(answer="Success1", confidence=9, reasoning="Reason1", citations=[])
    result2 = None
    result3 = DummyResponse(answer="Success3", confidence=7, reasoning="Reason3", citations=[1])
    # Add a 4th result for the judge call (expecting IntResponseModel)
    judge_result = IntResponseModel(answer=0, confidence=10, reasoning="Judge chose 0", citations=[]) # Judge selects the first candidate (index 0)
    mock_parent_generate.side_effect = [result1, result2, result3, judge_result]

    keys = ["key1", "key2", "key3"]
    # expected_results_list needs to reflect the structure before judge selection
    expected_results_list = [result1, result2, result3] 

    # Patch llm_configs lookup used by the service
    with patch("app.services.llm.pydantic_llm_service.llm_configs", mock_llm_configs):
        final_result = await multi_service.generate_completion(
            prompt="Test prompt",
            response_model=DummyResponse,
            llm_config_keys=keys
        )

    # Final result should be the one selected by the judge (index 0 -> result1)
    assert final_result == result1 
    assert hasattr(final_result, 'all_responses')
    # all_responses should contain the original results *before* judging
    assert final_result.all_responses == expected_results_list
    # The mock should now be called 4 times (3 parallel + 1 judge)
    assert mock_parent_generate.call_count == len(keys) + 1
    # Check calls were made with correct overrides
    mock_parent_generate.assert_any_call(prompt="Test prompt", response_model=DummyResponse, llm_config_override=mock_llm_configs["key1"])
    mock_parent_generate.assert_any_call(prompt="Test prompt", response_model=DummyResponse, llm_config_override=mock_llm_configs["key2"])
    mock_parent_generate.assert_any_call(prompt="Test prompt", response_model=DummyResponse, llm_config_override=mock_llm_configs["key3"])

@pytest.mark.asyncio
async def test_multi_completion_later_success(
    multi_service: PydanticMultiCompletionService,
    mock_llm_configs: dict[str, LLMConfig],
    mocker: MagicMock
):
    """Test successful completion where a later LLM call succeeds."""
    mock_parent_generate = mocker.patch(
        "app.services.llm.pydantic_llm_service.PydanticCompletionService.generate_completion",
        new_callable=AsyncMock
    )

    # Define results for each call
    result1 = None
    result2 = DummyResponse(answer="Success2", confidence=8, reasoning="Reason2", citations=[])
    result3 = DummyResponse(answer="Success3", confidence=7, reasoning="Reason3", citations=[2]) # Another success
    # Add a 4th result for the judge call (expecting IntResponseModel)
    # Judge sees [result2, result3], selects index 0 (result2)
    judge_result = IntResponseModel(answer=0, confidence=10, reasoning="Judge chose 0", citations=[])
    mock_parent_generate.side_effect = [result1, result2, result3, judge_result]

    keys = ["key1", "key2", "key3"]
    # expected_results_list needs to reflect the structure before judge selection
    expected_results_list = [result1, result2, result3]

    # Patch llm_configs lookup used by the service
    with patch("app.services.llm.pydantic_llm_service.llm_configs", mock_llm_configs):
        final_result = await multi_service.generate_completion(
            prompt="Test prompt",
            response_model=DummyResponse,
            llm_config_keys=keys
        )

    # Final result should be the one selected by the judge (index 0 -> result2)
    assert final_result == result2 
    assert hasattr(final_result, 'all_responses')
    # all_responses should contain the original results *before* judging
    assert final_result.all_responses == expected_results_list
    # The mock should now be called 4 times (3 parallel + 1 judge)
    assert mock_parent_generate.call_count == len(keys) + 1
    # Check calls were made with correct overrides
    mock_parent_generate.assert_any_call(prompt="Test prompt", response_model=DummyResponse, llm_config_override=mock_llm_configs["key1"])
    mock_parent_generate.assert_any_call(prompt="Test prompt", response_model=DummyResponse, llm_config_override=mock_llm_configs["key2"])
    mock_parent_generate.assert_any_call(prompt="Test prompt", response_model=DummyResponse, llm_config_override=mock_llm_configs["key3"])

@pytest.mark.asyncio
async def test_multi_completion_all_fail(
    multi_service: PydanticMultiCompletionService,
    mock_llm_configs: dict[str, LLMConfig],
    mocker: MagicMock
):
    """Test completion where all LLM calls fail (return None or exception)."""
    mock_parent_generate = mocker.patch(
        "app.services.llm.pydantic_llm_service.PydanticCompletionService.generate_completion",
        new_callable=AsyncMock
    )

    # Define results for each call
    result1 = None
    result2 = None
    # Simulate an exception from the underlying call
    result3_exception = ValueError("LLM API Error")
    mock_parent_generate.side_effect = [result1, result2, result3_exception]

    keys = ["key1", "key2", "key3"]
    # Exception in gather results is caught, processed result becomes None
    expected_results_list = [result1, result2, None]

    # Patch llm_configs lookup used by the service
    with patch("app.services.llm.pydantic_llm_service.llm_configs", mock_llm_configs):
        final_result = await multi_service.generate_completion(
            prompt="Test prompt",
            response_model=DummyResponse,
            llm_config_keys=keys
        )

    assert final_result is None # Should return None as no call succeeded
    # We cannot check final_result.all_responses as final_result is None
    assert mock_parent_generate.call_count == len(keys)

@pytest.mark.asyncio
async def test_multi_completion_invalid_key(
    multi_service: PydanticMultiCompletionService,
    mock_llm_configs: dict[str, LLMConfig],
    mocker: MagicMock
):
    """Test completion with a mix of valid and invalid keys."""
    mock_parent_generate = mocker.patch(
        "app.services.llm.pydantic_llm_service.PydanticCompletionService.generate_completion",
        new_callable=AsyncMock
    )

    # Define results only for the valid keys
    result1 = DummyResponse(answer="Success1", confidence=9, reasoning="Reason1", citations=[])
    result3 = None # Third key (key3) is valid but returns None
    mock_parent_generate.side_effect = [result1, result3] # Only two actual calls expected

    keys = ["key1", "invalid_key", "key3"]
    # Expected list reflects placeholders for invalid keys
    expected_results_list = [result1, None, result3]

    # Patch llm_configs lookup used by the service
    with patch("app.services.llm.pydantic_llm_service.llm_configs", mock_llm_configs):
        final_result = await multi_service.generate_completion(
            prompt="Test prompt",
            response_model=DummyResponse,
            llm_config_keys=keys
        )

    assert final_result == result1 # Should select the first non-None
    assert hasattr(final_result, 'all_responses')
    # all_responses includes None for the invalid key position
    assert final_result.all_responses == expected_results_list
    # Parent method only called for valid keys
    assert mock_parent_generate.call_count == 2
    mock_parent_generate.assert_any_call(prompt="Test prompt", response_model=DummyResponse, llm_config_override=mock_llm_configs["key1"])
    mock_parent_generate.assert_any_call(prompt="Test prompt", response_model=DummyResponse, llm_config_override=mock_llm_configs["key3"])

@pytest.mark.asyncio
async def test_multi_completion_default_keys(
    multi_service: PydanticMultiCompletionService,
    mock_llm_configs: dict[str, LLMConfig],
    mocker: MagicMock
):
    """Test completion using the default list of keys."""
    mock_parent_generate = mocker.patch(
        "app.services.llm.pydantic_llm_service.PydanticCompletionService.generate_completion",
        new_callable=AsyncMock
    )
    # Actual default keys from the service code
    actual_default_keys = [
        "o4-mini",
        "gemini-2.5-flash",
        "gpt-4.1-mini",
        "gpt-4.1-nano"
    ]
    num_actual_defaults = len(actual_default_keys)

    # Simulate results ONLY for the keys the service will ACTUALLY call (based on logs/validity)
    # Logs showed o4-mini & gpt-4.1-nano are skipped (invalid)
    # Logs showed gemini-2.5-flash call resulted in None
    # Logs showed gpt-4.1-mini call succeeded
    mock_result_gemini_flash = None
    mock_result_gpt41_mini = DummyResponse(answer="SuccessMini", confidence=9, reasoning="ReasonMini", citations=[])
    mock_parent_generate.side_effect = [
        mock_result_gemini_flash, # For gemini-2.5-flash call
        mock_result_gpt41_mini  # For gpt-4.1-mini call
    ]

    # Expected results in all_responses corresponds to the *order* of actual_default_keys
    expected_results_list = [
        None,                      # o4-mini (skipped)
        mock_result_gemini_flash,  # gemini-2.5-flash
        mock_result_gpt41_mini,    # gpt-4.1-mini
        None                       # gpt-4.1-nano (skipped)
    ]

    # Patch llm_configs lookup used by the service
    with patch("app.services.llm.pydantic_llm_service.llm_configs", mock_llm_configs):
        # Call without specifying llm_config_keys
        final_result = await multi_service.generate_completion(
            prompt="Test prompt",
            response_model=DummyResponse
        )

    # Only one success (gpt-4.1-mini), so that should be the result
    assert final_result == mock_result_gpt41_mini
    assert hasattr(final_result, 'all_responses')
    assert final_result.all_responses == expected_results_list
    # Mock should only be called for the valid keys
    assert mock_parent_generate.call_count == 2 

    # Verify the specific calls made
    # (Order depends on gather, so use assert_any_call)
    mock_parent_generate.assert_any_call(
        prompt="Test prompt", 
        response_model=DummyResponse, 
        llm_config_override=mock_llm_configs.get("gemini-2.5-flash")
    )
    mock_parent_generate.assert_any_call(
        prompt="Test prompt", 
        response_model=DummyResponse, 
        llm_config_override=mock_llm_configs.get("gpt-4.1-mini")
    )

@pytest.mark.asyncio
async def test_multi_completion_single_key_compatibility(
    multi_service: PydanticMultiCompletionService,
    mock_llm_configs: dict[str, LLMConfig],
    mocker: MagicMock
):
    """Test compatibility when called with a single key, mimicking parent override."""
    mock_parent_generate = mocker.patch(
        "app.services.llm.pydantic_llm_service.PydanticCompletionService.generate_completion",
        new_callable=AsyncMock
    )

    # Define result for the single call
    key_to_use = "key2"
    config_to_use = mock_llm_configs[key_to_use]
    expected_parent_result = DummyResponse(answer="SingleSuccess", confidence=10, reasoning="SingleReason", citations=[])
    mock_parent_generate.return_value = expected_parent_result # Only one call expected

    expected_results_list = [expected_parent_result]

    # Patch llm_configs lookup used by the service
    with patch("app.services.llm.pydantic_llm_service.llm_configs", mock_llm_configs):
        final_result = await multi_service.generate_completion(
            prompt="Test prompt single",
            response_model=DummyResponse,
            llm_config_keys=[key_to_use] # Pass only one key
        )

    # Result should match the parent's output for that config
    assert final_result.answer == expected_parent_result.answer
    assert final_result.confidence == expected_parent_result.confidence
    assert final_result.reasoning == expected_parent_result.reasoning
    assert final_result.__class__ == expected_parent_result.__class__

    # Verify all_responses is also present and correct
    assert hasattr(final_result, 'all_responses')
    assert final_result.all_responses == expected_results_list

    # Ensure parent was called exactly once with the correct override
    mock_parent_generate.assert_called_once_with(
        prompt="Test prompt single",
        response_model=DummyResponse,
        llm_config_override=config_to_use
    )

# Add more tests here following the plan... 