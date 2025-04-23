"""Pydantic AI completion service implementation."""

import logging
from typing import Any, Optional, Type

from pydantic import BaseModel
from pydantic_ai import Agent

# Import the configuration function
from app.services.llm.llm_configuration import get_active_llm_config

from app.core.config import Settings
from app.services.llm.base import CompletionService
from app.models.llm_responses import SubQueriesResponseModel # Added for decompose_query matching

logger = logging.getLogger(__name__)


class PydanticCompletionService(CompletionService):
    """Pydantic AI completion service implementation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # No need to check API keys/model here, config loader handles it.
        # Agent is no longer initialized here
        logger.info("PydanticCompletionService initialized. LLM configuration will be loaded on demand.")

    async def generate_completion(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> Optional[BaseModel]:
        """Generate a completion using Pydantic AI Agent."""
        # Removed redundant settings checks, config loader handles errors.

        try:
            # Get the currently active LLM configuration
            active_config = get_active_llm_config()
            llm_instance = active_config.llm
            llm_settings = active_config.settings

            if not llm_instance:
                 logger.error("Failed to load LLM instance from configuration.")
                 return None

            # Instantiate the agent with the loaded LLM instance and response model
            agent = Agent(llm_instance, output_type=response_model, instrument=True)

            print(f"Calling agent.run with LLM: {llm_instance} and settings: {llm_settings}")
            # Pass the settings from the config as keyword arguments to run
            # Note: Ensure the keys in llm_settings match valid arguments for the specific pydantic-ai model's run/generation method (e.g., max_tokens)
            response = await agent.run(prompt, **llm_settings)

            logger.info(f"Raw response from agent.run: {response}")

            # Based on logs, agent.run returns AgentRunResult(output=PydanticModel).
            # Access the actual Pydantic model via the .output attribute.
            # Let AttributeError propagate if the structure changes unexpectedly.
            actual_response = response.output

            if actual_response is None:
                 logger.warning("Received None response from Pydantic AI Agent or extracted output is None")
                 return None

            if not isinstance(actual_response, response_model):
                 logger.error(f"Extracted response is not of the expected type {response_model.__name__}. Got {type(actual_response).__name__}")
                 return None

            return actual_response
        except Exception as e:
            logger.error(f"Error generating completion with Pydantic AI: {e}", exc_info=True)
            return None

    async def decompose_query(self, query: str) -> dict[str, Any]:
        """Decompose the query into smaller sub-queries (placeholder)."""
        # Removed check for self.agent
        # TODO: Implement actual decomposition using Pydantic AI Agent
        # This would involve getting the active config and instantiating an Agent
        # with SubQueriesResponseModel here, similar to generate_completion.
        # active_config = get_active_llm_config()
        # agent = Agent(active_config.llm, output_type=SubQueriesResponseModel, instrument=True)
        # response = await agent.run(f"Decompose this query: {query}", **active_config.settings)
        # return response.output.model_dump() # Assuming SubQueriesResponseModel has a field like `sub_queries`

        logger.warning("Decomposition not implemented for Pydantic AI, returning original query.")
        return {"sub_queries": [query]} 