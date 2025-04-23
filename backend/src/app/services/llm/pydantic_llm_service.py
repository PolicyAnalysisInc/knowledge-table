"""Pydantic AI completion service implementation."""

import asyncio
import logging
from typing import Any, List, Optional, Type, Dict

from pydantic import BaseModel, create_model
from pydantic_core import PydanticUndefined
from pydantic_ai import Agent

# Import the configuration function and type AND the configs dictionary
from app.services.llm.llm_configuration import LLMConfig, get_active_llm_config, llm_configs

from app.core.config import Settings
from app.services.llm.base import CompletionService
from app.models.llm_responses import BaseResponseModel, SubQueriesResponseModel # Added for decompose_query matching

logger = logging.getLogger(__name__)


class PydanticCompletionService(CompletionService):
    """Pydantic AI completion service implementation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # No need to check API keys/model here, config loader handles it.
        # Agent is no longer initialized here
        logger.info("PydanticCompletionService initialized. LLM configuration will be loaded on demand.")

    async def generate_completion(
        self,
        prompt: str,
        response_model: Type[BaseResponseModel],
        llm_config_override: Optional[LLMConfig] = None,
    ) -> Optional[BaseResponseModel]:
        """
        Generate a completion using Pydantic AI Agent.

        Args:
            prompt: The input prompt for the LLM.
            response_model: The Pydantic model to structure the response.
            llm_config_override: Optional LLMConfig to override the active configuration.

        Returns:
            An instance of the response_model or None if generation fails or is empty.
        """
        # Removed redundant settings checks, config loader handles errors.

        try:
            # Get the currently active LLM configuration parameters, or use the override
            active_config = llm_config_override or get_active_llm_config()
            # Instantiate the LLM model on demand
            llm_instance = active_config.get_instance()
            llm_settings = active_config.settings

            # --- Add detailed logging here ---
            logger.info(f"[generate_completion] Using config: provider='{active_config.provider}', model='{active_config.model_name}', settings={llm_settings}")
            # --- End logging ---

            if not llm_instance:
                 # The get_instance method raises errors now, so this might be redundant,
                 # but keep it for safety or if get_instance is changed later.
                 logger.error("Failed to load LLM instance from configuration.")
                 return None

            # Dynamically create a temporary model for the agent, excluding 'all_responses'
            agent_model_fields = {
                k: (v.annotation, v.default if v.default is not PydanticUndefined else ...)
                for k, v in response_model.model_fields.items()
                if k != 'all_responses'
            }
            AgentResponseType = create_model(
                f"{response_model.__name__}ForAgent",
                **agent_model_fields,
                __base__=BaseModel # Inherit from plain BaseModel
            )

            # Instantiate the agent with the temporary model type
            instrument_name = f"Agent-{active_config.provider}-{active_config.model_name}"
            agent = Agent(
                llm_instance,
                output_type=AgentResponseType, # Use the temporary model
                instrument=True,
                name=instrument_name
            )

            print(f"Calling agent.run ({instrument_name}) with LLM: {llm_instance} and settings: {llm_settings}")
            response = await agent.run(prompt, **llm_settings)
            logger.info(f"Raw response from agent.run: {response}")

            # Get the result (instance of AgentResponseType or None)
            agent_result: Optional[AgentResponseType] = response.output

            if agent_result is None:
                 logger.warning("Received None response from Pydantic AI Agent or extracted output is None")
                 return None

            # Convert the agent result back to the original response_model type
            try:
                final_result = response_model(**agent_result.model_dump())
            except Exception as conversion_e:
                 logger.error(f"Failed to convert agent result back to {response_model.__name__}: {conversion_e}", exc_info=True)
                 return None # Failed conversion means we can't return the required type

            # Check if all fields (excluding metadata/all_responses) in the *original* model are None
            # Need to re-evaluate this check logic if necessary
            final_dump = final_result.model_dump(exclude={'all_responses'})
            if all(value is None for value in final_dump.values()):
                 logger.info(f"All core fields in the {response_model.__name__} response model are None. Returning None.")
                 return None

            return final_result # Return instance of original response_model type
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
        # llm_instance = active_config.get_instance() # Get instance here
        # agent = Agent(llm_instance, output_type=SubQueriesResponseModel, instrument=True)
        # response = await agent.run(f"Decompose this query: {query}", **active_config.settings)
        # return response.output.model_dump() # Assuming SubQueriesResponseModel has a field like `sub_queries`

        logger.warning("Decomposition not implemented for Pydantic AI, returning original query.")
        return {"sub_queries": [query]} 


# === PydanticMultiCompletionService ===

class PydanticMultiCompletionService(PydanticCompletionService):
    """
    A service that overrides PydanticCompletionService to run completions in parallel
    and selects the first successful result, populating the 'all_responses' field.
    """
    def __init__(self, settings: Settings):
        """
        Initializes the PydanticMultiCompletionService.

        Args:
            settings: The application settings.
        """
        super().__init__(settings)
        logger.info("PydanticMultiCompletionService initialized.")

    # Override the parent method
    async def generate_completion(
        self,
        prompt: str,
        response_model: Type[BaseResponseModel],
        llm_config_keys: Optional[List[str]] = None, # Default to None
    ) -> Optional[BaseResponseModel]:
        """
        Overrides the parent method to generate multiple completions in parallel
        using different LLM configurations specified by llm_config_keys.
        It selects the first successful one, populates its 'all_responses' field,
        and returns it.

        Args:
            prompt: The input prompt for the LLM.
            response_model: The Pydantic model class inheriting from BaseResponseModel.
            llm_config_keys: A list of string keys identifying the LLMConfig to use
                             for each parallel run (from llm_configuration.llm_configs).
                             Defaults to ["gemini-2.5", "gemini-2.5-flash", "gpt-4o", "gpt-4.1-mini"]
                             if None is provided.

        Returns:
            The first non-None Pydantic model instance from the parallel calls,
            with its `all_responses` attribute populated. Returns None if all calls fail.
        """
        # Set default list if None is provided
        if llm_config_keys is None:
            llm_config_keys = [
                "gemini-2.5",
                "gemini-2.5-flash",
                "gpt-4o",
                "gpt-4.1-mini"
            ]

        num_runs = len(llm_config_keys)
        if num_runs == 0:
            logger.warning("generate_completion called with empty llm_config_keys list.")
            return None

        logger.info(f"Starting {num_runs} parallel completions for prompt: '{prompt[:50]}...' with configs: {llm_config_keys}")

        tasks = []
        valid_keys_configs = [] # Keep track of which configs were used for results length

        # Create tasks, looking up config for each key
        for key in llm_config_keys:
            config = llm_configs.get(key)
            if not config:
                logger.error(f"Invalid LLM configuration key '{key}' provided. Skipping this run.")
                # Add a placeholder task that returns None immediately? Or just skip?
                # Skipping means the results list length might not match keys length if keys are invalid.
                # Let's add a placeholder task for consistent output length.
                async def none_task(): return None
                tasks.append(none_task())
                valid_keys_configs.append(None) # Placeholder for config
                continue

            valid_keys_configs.append(config) # Store the config used
            tasks.append(
                super().generate_completion(
                    prompt=prompt,
                    response_model=response_model,
                    llm_config_override=config, # Use the specific config for this task
                )
            )

        # Initialize results based on the number of keys provided
        results: List[Optional[BaseResponseModel]] = [None] * len(llm_config_keys)
        try:
            # Run tasks concurrently and gather results
            gathered_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results, separating successful models from exceptions/Nones
            processed_results: List[Optional[BaseResponseModel]] = []
            for i, res in enumerate(gathered_results):
                config_used = valid_keys_configs[i] # Get config used for this index
                key_used = llm_config_keys[i]
                log_prefix = f"Task {i} (key: '{key_used}', config: {config_used.model_name if config_used else 'InvalidKey'}) for prompt '{prompt[:50]}...'"

                if isinstance(res, Exception):
                    logger.error(f"{log_prefix} failed with exception: {res}", exc_info=res)
                    processed_results.append(None)
                elif res is None:
                     logger.warning(f"{log_prefix} returned None.")
                     processed_results.append(None)
                elif isinstance(res, BaseResponseModel):
                     processed_results.append(res)
                else:
                    # Log error for unexpected return types (e.g., from none_task)
                    if res is not None: # Avoid logging expected None from invalid key placeholder
                         logger.error(f"{log_prefix} returned unexpected type: {type(res)}")
                    processed_results.append(None)
            results = processed_results # Update results with processed list

        except Exception as e:
            # Catch potential errors during the gather setup itself
            logger.error(
                f"Unexpected error during parallel completion gather setup for prompt '{prompt[:50]}...': {e}",
                exc_info=True
            )
            # results remains [None] * num_runs

        # Find the first non-None result from the processed list
        first_result: Optional[BaseResponseModel] = next((res for res in results if res is not None), None)

        if first_result:
            logger.info(f"Selected first successful result for prompt: '{prompt[:50]}...'")
            # Assign the full list of responses to the field in the chosen result
            first_result.all_responses = results
            logger.debug(f"Assigned 'all_responses' field to the chosen result.")
        else:
            logger.warning(f"All {num_runs} completion attempts failed or returned None for prompt: '{prompt[:50]}...'")

        # Return the chosen result (type Optional[BaseResponseModel], with populated all_responses) or None
        return first_result 