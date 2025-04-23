"""Factory for creating language model completion services."""

import logging
from typing import Optional
import os

from app.core.config import Settings
from app.services.llm.base import CompletionService
from app.services.llm.openai_llm_service import OpenAICompletionService
from app.services.llm.pydantic_llm_service import (
    PydanticCompletionService,
    PydanticMultiCompletionService,
)

logger = logging.getLogger(__name__)


class CompletionServiceFactory:
    """Factory for creating completion services."""

    @staticmethod
    def create_service(settings: Settings) -> Optional[CompletionService]:
        """Create a completion service based on the LLM_INTERFACE setting."""
        # Read the interface setting from environment variable or settings object
        # Prioritize environment variable
        interface_env = os.getenv("LLM_INTERFACE")
        if interface_env:
            interface = interface_env.lower()
            logger.info(f"Using LLM_INTERFACE from environment: '{interface}'")
        else:
            # Fallback to settings object, default to 'openai' if not set or empty
            interface = getattr(settings, 'llm_interface', 'openai').lower()
            logger.info(f"Using LLM_INTERFACE from settings: '{interface}'")

        logger.info(f"Attempting to create completion service for interface: {interface}")

        if interface == "openai":
            return OpenAICompletionService(settings)
        elif interface == "pydantic":
            return PydanticCompletionService(settings)
        elif interface == "pydantic-multi": # Changed from "multi"
            logger.info("Instantiating PydanticMultiCompletionService based on LLM_INTERFACE='pydantic-multi'.")
            return PydanticMultiCompletionService(settings)
        # Add more providers here when needed
        logger.warning(f"Unsupported LLM interface specified: '{interface}'. Returning None.")
        return None
