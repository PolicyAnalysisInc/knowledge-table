import os
import json # Ensure json is imported if using SERVICE_ACCOUNT_INFO
from pydantic import BaseModel, Field # Add necessary imports for the class
from typing import Any, Dict, Type # Add necessary imports for the class
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel

# Assuming default providers which read API keys from env vars
# (e.g., GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY)
# Or configure providers explicitly if needed:
# from pydantic_ai.providers import GoogleGLAProvider, AnthropicProvider, OpenAIProvider

# --- LLM Provider Class Mapping --- #
# Maps provider strings (used in config) to pydantic-ai model classes
PROVIDER_CLASS_MAP: Dict[str, Type[Any]] = {
    "google-gla": GeminiModel,
    "openai": OpenAIModel,
    "anthropic": AnthropicModel,
    # Add other provider mappings as needed
}

class LLMConfig(BaseModel):
    """Configuration for a language model, storing parameters for lazy instantiation."""
    model_name: str  # The specific model name (e.g., "gemini-2.5-pro-preview-03-25")
    provider: str    # The provider identifier (e.g., "google-gla")
    settings: Dict[str, Any] = Field(default_factory=dict) # Model-specific settings (e.g., max_tokens)

    def get_instance(self) -> Any:
        """Instantiates and returns the configured pydantic-ai model."""
        ModelClass = PROVIDER_CLASS_MAP.get(self.provider)
        if not ModelClass:
            raise ValueError(f"Unsupported provider '{self.provider}' configured.")
        
        try:
            if self.provider == "google-gla":
                 return ModelClass(model_name=self.model_name, provider=self.provider)
            else:
                 return ModelClass(model_name=self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate LLM {self.model_name} for provider {self.provider}: {e}") from e

# Removed default/fallback keys: USER_DEFAULT_LLM_CONFIG_KEY, FALLBACK_DEFAULT_LLM_CONFIG_KEY

# Define the dictionary of LLM configurations using parameters
llm_configs: dict[str, LLMConfig] = {
    # Gemini Models
    "gemini-2.5": LLMConfig(
        model_name="gemini-2.5-pro-preview-03-25", 
        provider="google-gla", 
        settings={}
    ),
    "gemini-2.5-flash": LLMConfig(
        model_name="gemini-2.5-flash-preview-04-17", 
        provider="google-gla", 
        settings={}
    ),

    # OpenAI Models
    "o1": LLMConfig(model_name="o1", provider="openai", settings={}),
    "o3": LLMConfig(model_name="o3", provider="openai", settings={}),
    "o3-mini": LLMConfig(model_name="o3-mini", provider="openai", settings={}),
    "o4-mini": LLMConfig(model_name="gpt-4o-mini", provider="openai", settings={}), # Assuming gpt-4o-mini
    "o1-high": LLMConfig(model_name="o1", provider="openai", settings={"reasoning_effort": "high"}),
    "o3-high": LLMConfig(model_name="o3", provider="openai", settings={"reasoning_effort": "high"}),
    "o3-mini-high": LLMConfig(model_name="o3-mini", provider="openai", settings={"reasoning_effort": "high"}),
    "o4-mini-high": LLMConfig(model_name="gpt-4o-mini", provider="openai", settings={"reasoning_effort": "high"}), # Assuming gpt-4o-mini
    "gpt-4.1": LLMConfig(model_name="gpt-4.1", provider="openai", settings={"max_tokens": 120000}),
    "gpt-4.1-mini": LLMConfig(model_name="gpt-4.1-mini-2025-04-14", provider="openai", settings={}),
    "gpt-4.1-nano": LLMConfig(model_name="gpt-4.1-nano-2025-04-14", provider="openai", settings={}),
    "gpt-4o": LLMConfig(model_name="gpt-4o", provider="openai", settings={"max_tokens": 60000}),

    # Anthropic Models
    "claude-3-5-sonnet": LLMConfig(
        model_name='claude-3-5-sonnet-20241022', 
        provider="anthropic", 
        settings={"max_tokens": 128000}
    ),
    "claude-3-7-sonnet": LLMConfig(
        model_name='claude-3-7-sonnet-20250219', 
        provider="anthropic", 
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
    Retrieves the active LLM configuration parameters based *only* on the LLM_CONFIG_KEY environment variable.
    Note: This returns the configuration parameters, not an instantiated model.

    Returns:
        The selected LLMConfig instance containing parameters.

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
        return config
    else:
        raise KeyError(
            f"LLM_CONFIG_KEY '{config_key}' is invalid. "
            "Available keys are: "
            f"{list(llm_configs.keys())}"
        ) 