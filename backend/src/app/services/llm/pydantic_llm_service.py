"""Pydantic AI completion service implementation."""

import asyncio
import logging
import json
from typing import Any, List, Optional, Type, Dict, Tuple

from pydantic import BaseModel, create_model
from pydantic_core import PydanticUndefined
from pydantic_ai import Agent

# Import the configuration function and type AND the configs dictionary
from app.services.llm.llm_configuration import LLMConfig, get_active_llm_config, llm_configs
# Import the new judge prompt
from app.services.llm.openai_prompts import JUDGE_RESPONSE_PROMPT

from app.core.config import Settings
from app.services.llm.base import CompletionService
from app.models.llm_responses import BaseResponseModel, SubQueriesResponseModel, IntResponseModel

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

            # Set the model name on the final result object
            final_result.model_name = active_config.model_name

            # Check if all fields (excluding metadata/all_responses) in the *original* model are None
            # Need to re-evaluate this check logic if necessary
            final_dump = final_result.model_dump(exclude={'all_responses', 'is_selected_answer', 'model_name'}) # Exclude metadata
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
    A service that overrides PydanticCompletionService to run completions in parallel,
    uses an LLM judge to select the best response, and populates the 'all_responses' field.
    """
    def __init__(self, settings: Settings):
        """
        Initializes the PydanticMultiCompletionService.

        Args:
            settings: The application settings.
        """
        super().__init__(settings)
        logger.info("PydanticMultiCompletionService initialized.")

    async def _select_best_response_with_judge(
        self,
        original_prompt: str,
        successful_responses_with_info: List[Tuple[int, BaseResponseModel, str, Optional[LLMConfig]]],
        successful_response_map: Dict[int, Tuple[int, BaseResponseModel, str, Optional[LLMConfig]]]
    ) -> Tuple[Optional[BaseResponseModel], str]:
        """
        Uses an LLM judge to select the best response from multiple successful candidates.

        Args:
            original_prompt: The original prompt (including query and context) sent to the models.
            successful_responses_with_info: List of (original_index, response, key, config) tuples.
            successful_response_map: Dictionary mapping judge index to original info tuple.

        Returns:
            A tuple containing the selected BaseResponseModel (or None if judge fails)
            and a string describing the selection reason.
        """
        logger.info(f"Using LLM judge to select the best from {len(successful_responses_with_info)} successful responses.")
        selected_response: Optional[BaseResponseModel] = None
        selected_reason = "Judge selection process initiated."

        # Prepare candidates for the judge prompt
        judge_candidates = []
        for judge_idx, (original_idx, response, key, config) in enumerate(successful_responses_with_info):
            judge_candidates.append({
                "index": judge_idx, # 0-based index for the judge
                "model_key": key,
                "response": response.model_dump(exclude={'all_responses'}) # Exclude nested field
            })

        # Format the judge prompt using the template
        judge_prompt = JUDGE_RESPONSE_PROMPT.substitute(
            original_query=original_prompt,
            candidate_responses_json=json.dumps(judge_candidates, indent=2)
        )

        try:
            judge_config = get_active_llm_config() # Use default active config for judge
            logger.info(f"Calling judge LLM (config: {judge_config.provider}/{judge_config.model_name}) to choose best response.")

            # Use super().generate_completion for the judge call
            judge_llm_response: Optional[IntResponseModel] = await super().generate_completion(
                prompt=judge_prompt,
                response_model=IntResponseModel, # Expecting {"answer": index}
                llm_config_override=judge_config
            )

            if judge_llm_response and judge_llm_response.answer is not None:
                chosen_judge_index = judge_llm_response.answer

                if chosen_judge_index in successful_response_map:
                    original_index, response, key, config = successful_response_map[chosen_judge_index]
                    selected_response = response
                    selected_reason = f"Judge selected response from index {chosen_judge_index} (original key: '{key}')."
                    logger.info(selected_reason)
                else:
                    logger.warning(f"Judge returned invalid index {chosen_judge_index}. Falling back to first successful response.")
                    # Fallback logic: Select the first successful response
                    original_index, response, key, config = successful_responses_with_info[0]
                    selected_response = response
                    selected_reason = f"Judge returned invalid index. Selected first successful response (key: '{key}')."
            else:
                logger.warning("Judge LLM call failed or returned invalid data. Falling back to first successful response.")
                # Fallback logic: Select the first successful response
                original_index, response, key, config = successful_responses_with_info[0]
                selected_response = response
                selected_reason = f"Judge call failed. Selected first successful response (key: '{key}')."

        except Exception as judge_e:
            logger.error(f"Error during judge LLM call: {judge_e}", exc_info=True)
            # Fallback logic: Select the first successful response
            original_index, response, key, config = successful_responses_with_info[0]
            selected_response = response
            selected_reason = f"Exception during judge call. Selected first successful response (key: '{key}')."

        return selected_response, selected_reason

    # Override the parent method
    async def generate_completion(
        self,
        prompt: str,
        response_model: Type[BaseResponseModel],
        llm_config_keys: Optional[List[str]] = None, # Default to None
    ) -> Optional[BaseResponseModel]:
        """
        Overrides the parent method to generate multiple completions in parallel
        using different LLM configurations. It then uses an LLM judge to select
        the best response based on accuracy, reasoning, and confidence. The chosen
        response's 'all_responses' field is populated with all initial results.

        Args:
            prompt: The input prompt for the LLM.
            response_model: The Pydantic model class inheriting from BaseResponseModel.
            llm_config_keys: A list of string keys identifying the LLMConfig to use
                             for each parallel run (from llm_configuration.llm_configs).
                             Defaults to ["gemini-2.5", "gemini-2.5-flash", "gpt-4o", "gpt-4.1-mini"]
                             if None is provided.

        Returns:
            The best Pydantic model instance selected by the judge LLM,
            with its `all_responses` attribute populated. Returns None if all
            initial calls fail or if the judge process fails unexpectedly.
        """
        # Set default list if None is provided
        if llm_config_keys is None:
            llm_config_keys = [
                "o4-mini",
                "gemini-2.5-flash",
                "gpt-4.1-mini",
                "gpt-4.1-nano"
            ]

        num_keys = len(llm_config_keys)
        if num_keys == 0:
            logger.warning("generate_completion called with empty llm_config_keys list.")
            return None

        logger.info(f"Starting {num_keys} parallel completions for prompt: '{prompt[:50]}...' with configs: {llm_config_keys}")

        tasks = []
        config_map = {} # Map task index to config info {task_idx: (key, config)}

        # Create tasks, looking up config for each key
        for i, key in enumerate(llm_config_keys):
            config = llm_configs.get(key)
            if not config:
                logger.warning(f"Invalid LLM configuration key '{key}' provided. This run will be skipped.")
                # Create a task that immediately returns None for skipped/invalid keys
                async def skipped_task(): return None
                tasks.append(skipped_task())
                config_map[i] = (key, None) # Mark config as None for this index
                continue

            config_map[i] = (key, config) # Store key and config
            tasks.append(
                super().generate_completion(
                    prompt=prompt,
                    response_model=response_model,
                    llm_config_override=config,
                )
            )

        # Run tasks concurrently and gather results (including exceptions)
        try:
            gathered_results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as gather_e:
            logger.error(f"Unexpected error during asyncio.gather: {gather_e}", exc_info=True)
            gathered_results = [gather_e] * len(tasks) # Assume all failed if gather fails

        # --- Process Results ---
        all_results_list: List[Optional[BaseResponseModel]] = [None] * len(tasks)
        successful_responses_with_info = [] # List of tuples: (index, response, key, config)

        for i, res in enumerate(gathered_results):
            key, config = config_map[i]
            log_prefix = f"Task {i} (key: '{key}', config: {config.model_name if config else 'Skipped'})"

            if isinstance(res, Exception):
                logger.error(f"{log_prefix} failed with exception: {res}", exc_info=isinstance(res, BaseException))
                all_results_list[i] = None # Store None for errors
            elif isinstance(res, BaseResponseModel):
                logger.info(f"{log_prefix} succeeded.")
                all_results_list[i] = res # Store successful response
                successful_responses_with_info.append((i, res, key, config))
            else: # Includes None results from skipped tasks or failed completions
                 if config: # Only log warning if it wasn't an intentionally skipped task
                     logger.warning(f"{log_prefix} returned None or unexpected type: {type(res)}.")
                 all_results_list[i] = None # Store None

        # --- Select Best Response ---
        num_successful = len(successful_responses_with_info)
        selected_response: Optional[BaseResponseModel] = None
        selected_reason = "No successful responses."

        if num_successful == 0:
            logger.warning(f"All {len(tasks)} completion attempts failed or returned None for prompt: '{prompt[:50]}...'")
            return None # No responses to choose from

        elif num_successful == 1:
            original_index, response, key, config = successful_responses_with_info[0]
            selected_response = response
            selected_reason = f"Only one successful response (from key: '{key}')."
            logger.info(selected_reason)

        else:
            # --- LLM Judge Selection ---
            logger.info(f"Multiple ({num_successful}) successful responses obtained. Using LLM judge to select the best.")

            # Prepare candidates for the judge prompt
            judge_candidates = []
            successful_response_map = {} # Map judge index back to original info
            for judge_idx, (original_idx, response, key, config) in enumerate(successful_responses_with_info):
                judge_candidates.append({
                    "index": judge_idx, # 0-based index for the judge
                    "model_key": key,
                    "response": response.model_dump(exclude={'all_responses'}) # Exclude nested field
                })
                successful_response_map[judge_idx] = (original_idx, response, key, config)

            # Call the helper method to select the best response
            selected_response, selected_reason = await self._select_best_response_with_judge(
                original_prompt=prompt, # Pass original prompt
                successful_responses_with_info=successful_responses_with_info,
                successful_response_map=successful_response_map
            )

        # --- Finalize and Return ---
        if selected_response:
            # Set the flag on the chosen response object
            selected_response.is_selected_answer = True
            # Assign the full list (which includes the flagged response) back
            selected_response.all_responses = all_results_list
            logger.debug(f"Assigned 'all_responses' field and set 'is_selected_answer=True' on the chosen result. Selection reason: {selected_reason}")
            return selected_response
        else:
            # This case should ideally not be reached if num_successful > 0, but as a safeguard:
            logger.error("Failed to select a final response despite having successful candidates. Returning None.")
            return None

