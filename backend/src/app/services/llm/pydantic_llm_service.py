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
        if settings.openai_api_key and settings.llm_model:
            # Construct the model identifier string (e.g., "openai:gpt-4o")
            model_identifier = f"openai:{settings.llm_model}"
            # Initialize Agent with model identifier.
            # API key is expected to be picked up from OPENAI_API_KEY env var.
            self.agent = Agent(model_identifier)
        else:
            self.agent = None
            logger.warning(
                "OpenAI API key or LLM model not set in settings. "
                "Pydantic AI features require OPENAI_API_KEY environment variable to be set."
            )

    async def generate_completion(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> Optional[BaseModel]:
        """Generate a completion using Pydantic AI Agent."""
        if self.agent is None:
            logger.warning(
                "Pydantic AI agent is not initialized. Skipping generation."
            )
            return None

        try:
            # The 'run' method in pydantic-ai is synchronous in older versions,
            # but let's assume a potential async version or wrapper might exist.
            # If Agent.run is sync, this might need adjustment depending on library evolution
            # or running it in an executor. For now, we'll call it directly.
            # Assuming Agent handles the parsing and validation internally.
            response: BaseModel = await self.agent.run(prompt, output_model=response_model)

            logger.info(f"Generated response: {response}")

            if response is None:
                logger.warning("Received None response from Pydantic AI Agent")
                return None

            # Pydantic-ai agent should return an instance of response_model
            # We might still want the check for all None values like in the original service
            if all(
                value is None for value in response.model_dump().values()
            ):
                logger.info("All fields in the Pydantic AI response are None")
                return None

            return response
        except Exception as e:
            # Catching broad Exception as pydantic-ai might raise various errors
            logger.error(f"Error generating completion with Pydantic AI: {e}", exc_info=True)
            return None

    async def decompose_query(self, query: str) -> dict[str, Any]:
        """Decompose the query into smaller sub-queries (placeholder)."""
        if self.agent is None:
            logger.warning(
                "Pydantic AI agent is not initialized. Skipping decomposition."
            )
            # Match original behavior if not initialized
            return {"sub_queries": [query]}

        # TODO: Implement actual decomposition using Pydantic AI Agent
        # This might involve a specific prompt and the SubQueriesResponseModel
        # For now, replicating the placeholder behavior of OpenAICompletionService
        logger.info("Decomposition not implemented for Pydantic AI, returning original query.")
        return {"sub_queries": [query]} 