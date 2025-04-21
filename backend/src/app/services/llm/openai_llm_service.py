"""OpenAI completion service implementation."""

import logging
import json
from typing import Any, Optional, Type, List

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.services.llm.base import CompletionService
from app.models.llm_responses import CitedResponseWrapper

logger = logging.getLogger(__name__)


class OpenAICompletionService(CompletionService):
    """OpenAI completion service implementation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key)
        else:
            self.client = None  # type: ignore
            logger.warning(
                "OpenAI API key is not set. LLM features will be disabled."
            )

    async def generate_completion(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> Optional[BaseModel]:
        """Generate a completion from the language model."""
        if self.client is None:
            logger.warning(
                "OpenAI client is not initialized. Skipping generation."
            )
            return None

        response_content: Optional[str] = None
        try:
            logger.debug(f"Sending prompt to LLM: {prompt}")
            messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]

            response_format_arg = None
            if response_model == CitedResponseWrapper:
                response_format_arg = {"type": "json_object"}
                logger.debug("Requesting JSON object format from OpenAI.")

            response: ChatCompletion = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                response_format=response_format_arg,
                temperature=0.0,
            )

            response_content = response.choices[0].message.content
            logger.debug(f"Raw response content from OpenAI: {response_content}")

            if response_content is None:
                logger.warning("Received None content from OpenAI")
                return None

            parsed_data = json.loads(response_content)

            validated_response = response_model(**parsed_data)
            logger.info(f"Validated response: {validated_response.model_dump()}")

            if all(
                value is None
                for value in validated_response.model_dump().values()
            ):
                logger.info("All fields in the response are None")
            return validated_response

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response from OpenAI: {e}. Response content: {response_content}")
            return None
        except ValidationError as e:
            logger.error(f"Error validating parsed JSON against Pydantic model {response_model.__name__}: {e}. Parsed data: {parsed_data}")
            return None
        except Exception as e:
            logger.error(f"Error during LLM call or processing: {e}", exc_info=True)
            return None

    async def decompose_query(self, query: str) -> dict[str, Any]:
        """Decompose the query into smaller sub-queries."""
        if self.client is None:
            logger.warning(
                "OpenAI client is not initialized. Skipping decomposition."
            )
            return {"sub_queries": [query]}

        # TODO: Implement the actual decomposition logic here
        return {"sub_queries": [query]}
