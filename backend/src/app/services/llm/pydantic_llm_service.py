"""Pydantic AI completion service implementation."""

import logging
from typing import Any, Optional, Type

from pydantic import BaseModel
from pydantic_ai import Agent

from app.core.config import Settings
from app.services.llm.base import CompletionService
from app.models.llm_responses import SubQueriesResponseModel # Added for decompose_query matching

logger = logging.getLogger(__name__)


class PydanticCompletionService(CompletionService):
    """Pydantic AI completion service implementation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Agent is no longer initialized here
        if not settings.openai_api_key or not settings.llm_model:
             logger.warning(
                "OpenAI API key or LLM model not set in settings. "
                "Pydantic AI features require OPENAI_API_KEY environment variable to be set."
            )

    async def generate_completion(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> Optional[BaseModel]:
        """Generate a completion using Pydantic AI Agent."""
        if not self.settings.openai_api_key or not self.settings.llm_model:
            logger.warning(
                "OpenAI API key or LLM model not configured. Skipping Pydantic AI generation."
            )
            return None

        try:
            # Construct the model identifier string
            model_identifier = f"openai:{self.settings.llm_model}"

            # Instantiate the agent here with the specific response_model as output_type
            agent = Agent(model_identifier, output_type=response_model, instrument=True)

            print("Calling agent.run")
            # Call run without output_model, as it's set in Agent init
            response = await agent.run(prompt)

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
        if not self.settings.openai_api_key or not self.settings.llm_model:
             logger.warning(
                 "OpenAI API key or LLM model not configured. Skipping Pydantic AI decomposition."
             )
             return {"sub_queries": [query]}

        # TODO: Implement actual decomposition using Pydantic AI Agent
        # This would involve instantiating an Agent with SubQueriesResponseModel here.
        logger.info("Decomposition not implemented for Pydantic AI, returning original query.")
        return {"sub_queries": [query]} 