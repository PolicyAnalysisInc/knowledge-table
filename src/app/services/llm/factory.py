"""Factory for creating language model completion services."""

import logging
from typing import Optional

from app.core.config import Settings
from app.services.llm.base import CompletionService
from app.services.llm.openai_llm_service import OpenAICompletionService
from app.services.llm.pydantic_llm_service import PydanticCompletionService

logger = logging.getLogger(__name__)


class CompletionServiceFactory:
    """Factory for creating completion services."""

    @staticmethod
    def create_service(settings: Settings) -> Optional[CompletionService]:
        """Create a completion service based on the LLM_INTERFACE setting."""
        # Read the interface setting, default to 'openai' if not set or empty
        interface = settings.llm_interface.lower() if settings.llm_interface else "openai"
        logger.info(f"Attempting to create completion service for interface: {interface}")

        if interface == "pydantic":
            logger.info("Creating PydanticCompletionService.")
            return PydanticCompletionService(settings) # Pass settings
        elif interface == "openai":
            logger.info("Creating OpenAICompletionService.")
            return OpenAICompletionService(settings)
        else:
            # Log a warning if the specified interface is unsupported, but still default to OpenAI
            logger.warning(f"Unsupported LLM interface specified: '{settings.llm_interface}'. Defaulting to OpenAI.")
            return OpenAICompletionService(settings) 