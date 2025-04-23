import os
import json # Ensure json is imported if using SERVICE_ACCOUNT_INFO
from pydantic import BaseModel, Field # Add necessary imports for the class
from typing import Any, Dict # Add necessary imports for the class
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel

# Assuming default providers which read API keys from env vars
# (e.g., GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY)
# Or configure providers explicitly if needed:
# from pydantic_ai.providers import GoogleGLAProvider, AnthropicProvider, OpenAIProvider

# Define the configuration class directly in this file
class LLMConfig(BaseModel):
    """Configuration for a language model, including the model instance and settings."""
    llm: Any  # Can hold pydantic-ai model instances (GeminiModel, OpenAIModel, etc.) or potentially identifiers
    settings: Dict[str, Any] = Field(default_factory=dict)

# Removed default/fallback keys: USER_DEFAULT_LLM_CONFIG_KEY, FALLBACK_DEFAULT_LLM_CONFIG_KEY

# Define the dictionary of LLM configurations using user's exact definitions
llm_configs: dict[str, LLMConfig] = {
    # Gemini Models (Using exact names provided by user)
    "gemini-2.5": LLMConfig(
        llm=GeminiModel(model_name="gemini-2.5-pro-preview-03-25", provider="google-gla"),
        settings={}
    ),
    "gemini-2.5-flash": LLMConfig(
        llm=GeminiModel(model_name="gemini-2.5-flash-preview-04-17", provider="google-gla"),
        settings={}
    ),

    # OpenAI Models (Using exact names provided by user)
    "o1": LLMConfig(
        llm=OpenAIModel(model_name="o1"),
        settings={}
    ),
    "o3": LLMConfig(
        llm=OpenAIModel(model_name="o3"),
        settings={}
    ),
    "o3-mini": LLMConfig(
        llm=OpenAIModel(model_name="o3-mini"),
        settings={}
    ),
    "o4-mini": LLMConfig(
        # Assuming user means gpt-4o-mini here, as it's a common short form
        llm=OpenAIModel(model_name="gpt-4o-mini"), 
        settings={}
    ),
    "o1-high": LLMConfig(
        llm=OpenAIModel(model_name="o1"),
        settings={"reasoning_effort": "high"}
    ),
    "o3-high": LLMConfig(
        llm=OpenAIModel(model_name="o3"),
        settings={"reasoning_effort": "high"}
    ),
    "o3-mini-high": LLMConfig(
        llm=OpenAIModel(model_name="o3-mini"),
        settings={"reasoning_effort": "high"}
    ),
    "o4-mini-high": LLMConfig(
        llm=OpenAIModel(model_name="gpt-4o-mini"), # Assuming user means gpt-4o-mini
        settings={"reasoning_effort": "high"}
    ),
    "gpt-4.1": LLMConfig(
        llm=OpenAIModel(model_name="gpt-4.1"), # Using exact name
        settings={"max_tokens": 120000}
    ),
    "gpt-4.1-mini": LLMConfig(
        llm=OpenAIModel(model_name="gpt-4.1-mini-2025-04-14"), # Using exact name
        settings={}
    ),
    "gpt-4.1-nano": LLMConfig(
        llm=OpenAIModel(model_name="gpt-4.1-nano-2025-04-14"), # Using exact name
        settings={}
    ),
    "gpt-4o": LLMConfig(
        llm=OpenAIModel(model_name="gpt-4o"),
        settings={"max_tokens": 60000}
    ),

    # Anthropic Models (Using exact names provided by user)
    "claude-3-5-sonnet": LLMConfig(
        llm=AnthropicModel(model_name='claude-3-5-sonnet-20241022'),
        settings={"max_tokens": 128000}
    ),
    "claude-3-7-sonnet": LLMConfig(
        llm=AnthropicModel(model_name='claude-3-7-sonnet-20250219'),
        settings={
            "max_tokens": 128000,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 32000
            }
        }
    )
}

# Removed alias assignments: default, o4m, c3.5s

def get_active_llm_config() -> LLMConfig:
    """
    Retrieves the active LLM configuration based *only* on the LLM_CONFIG_KEY environment variable.

    Returns:
        The selected LLMConfig instance.

    Raises:
        ValueError: If the LLM_CONFIG_KEY environment variable is not set.
        KeyError: If the key specified in LLM_CONFIG_KEY is not found in llm_configs.
    """
    config_key = os.getenv("LLM_CONFIG_KEY")

    if not config_key:
        raise ValueError(
            "LLM_CONFIG_KEY environment variable is not set. "
            "Please set it to one of the keys defined in llm_configuration.py: "
            f"{list(llm_configs.keys())}"
        )

    config = llm_configs.get(config_key)

    if config:
        # Removed print statement
        return config
    else:
        # Raise KeyError if the key is set but invalid
        raise KeyError(
            f"LLM_CONFIG_KEY '{config_key}' is invalid. "
            "Available keys are: "
            f"{list(llm_configs.keys())}"
        ) 