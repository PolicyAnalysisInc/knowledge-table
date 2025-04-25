"""Query service."""

import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Union

from app.models.query_core import Chunk, FormatType, QueryType, Rule
from app.schemas.query_api import (
    QueryResult,
    ResolvedEntitySchema,
    SearchResponse,
)
from app.services.llm_service import (
    CompletionService,
    generate_inferred_response,
    generate_response,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SearchMethod = Callable[[str, str, List[Rule]], Awaitable[SearchResponse]]


def get_search_method(
    query_type: QueryType, vector_db_service: Any
) -> SearchMethod:
    """Get the search method based on the query type."""
    if query_type == "decomposition":
        return vector_db_service.decomposed_search
    elif query_type == "hybrid":
        return vector_db_service.hybrid_search
    else:  # simple_vector
        return lambda q, d, r: vector_db_service.vector_search([q], d)


def extract_chunks(search_response: SearchResponse) -> List[Chunk]:
    """Extract chunks from the search response."""
    return (
        search_response["chunks"]
        if isinstance(search_response, dict)
        else search_response.chunks
    )


def replace_keywords(
    text: Union[str, List[str]], keyword_replacements: Dict[str, str]
) -> tuple[
    Union[str, List[str]], Dict[str, Union[str, List[str]]]
]:  # Changed return type
    """Replace keywords in text and return both the modified text and transformation details."""
    if not text or not keyword_replacements:
        return text, {
            "original": text,
            "resolved": text,
        }  # Return dict instead of TransformationDict

    # Handle list of strings
    if isinstance(text, list):
        original_text = text.copy()
        result = []
        modified = False

        # Create a single regex pattern for all keywords
        pattern = "|".join(map(re.escape, keyword_replacements.keys()))
        regex = re.compile(f"\\b({pattern})\\b")

        for item in text:
            # Single pass replacement for all keywords
            new_item = regex.sub(
                lambda m: keyword_replacements[m.group()], item
            )
            result.append(new_item)
            if new_item != item:
                modified = True

        if modified:
            return result, {"original": original_text, "resolved": result}
        return result, {"original": original_text, "resolved": result}

    # Handle single string
    return replace_keywords_in_string(text, keyword_replacements)


def replace_keywords_in_string(
    text: str, keyword_replacements: Dict[str, str]
) -> tuple[str, Dict[str, Union[str, List[str]]]]:  # Changed return type
    """Keywords for single string."""
    if not text:
        return text, {"original": text, "resolved": text}

    # Create a single regex pattern for all keywords
    pattern = "|".join(map(re.escape, keyword_replacements.keys()))
    regex = re.compile(f"\\b({pattern})\\b")

    # Single pass replacement
    result = regex.sub(lambda m: keyword_replacements[m.group()], text)

    # Only return transformation if something changed
    if result != text:
        return result, {"original": text, "resolved": result}
    return text, {"original": text, "resolved": text}


async def process_query(
    query_type: QueryType,
    query: str,
    document_id: str,
    rules: List[Rule],
    format: FormatType,
    llm_service: CompletionService,
    vector_db_service: Any,
) -> QueryResult:
    """Process the query based on the specified type."""
    search_method = get_search_method(query_type, vector_db_service)

    # Prepare chunk content and identifiers for the LLM prompt
    search_response = await search_method(query, document_id, rules)
    chunks = extract_chunks(search_response)

    # Format chunks with index-based identifiers for the prompt: "[chunk_0] content0 [chunk_1] content1 ..."
    formatted_chunks_for_prompt = " ".join(
        f"[chunk_{i}] {chunk.content}" for i, chunk in enumerate(chunks)
    )

    # Call LLM service
    llm_output = await generate_response(
        llm_service, query, formatted_chunks_for_prompt, rules, format
    )
    # Extract results including citations and reasoning
    answer_value = llm_output["answer"]
    citations = llm_output["citations"]
    reasoning = llm_output.get("reasoning") # Use .get for safety

    # --- Keyword resolution logic (remains the same) ---
    # ... (rest of the function including keyword resolution and QueryResult instantiation) ...
    transformations: Dict[str, Union[str, List[str]]] = {
        "original": "",
        "resolved": "",
    }
    resolved_entities_output = None # Initialize resolved entities

    if format in ["str", "str_array"]:
        resolve_entity_rules = [
            rule for rule in rules if rule.type == "resolve_entity"
        ]
        replacements: Dict[str, str] = {}
        entity_type_map: Dict[str, str] = {} # Map resolved value to entity type

        if resolve_entity_rules and answer_value:
            for rule in resolve_entity_rules:
                if rule.options:
                    # Assume options are "original:resolved:entityType"
                    for option in rule.options:
                         parts = option.split(":")
                         if len(parts) == 3:
                              original, resolved, entityType = parts
                              replacements[original] = resolved
                              # Store the entity type associated with the *resolved* value
                              entity_type_map[resolved] = entityType
                         else:
                              logger.warning(f"Skipping invalid rule option format: {option}")

            if replacements:
                print(f"Resolving entities in answer: {answer_value}")
                transformed_value, transform_dict = replace_keywords(
                    answer_value, replacements
                )

                # Build ResolvedEntitySchema list
                resolved_entities_list = []
                original_list = transform_dict["original"]
                resolved_list = transform_dict["resolved"]

                # Ensure comparison happens between lists of same type/structure
                if isinstance(original_list, str): original_list = [original_list]
                if isinstance(resolved_list, str): resolved_list = [resolved_list]

                if isinstance(original_list, list) and isinstance(resolved_list, list) and len(original_list) == len(resolved_list):
                     for orig, res in zip(original_list, resolved_list):
                          if orig != res: # Only include if transformation occurred
                               entity_type = entity_type_map.get(res, "unknown") # Get entity type
                               resolved_entities_list.append(
                                   ResolvedEntitySchema(
                                       original=orig,
                                       resolved=res,
                                       # Assuming source is always 'column' for now, ID needs context
                                       source={"type": "column", "id": "unknown"},
                                       entityType=entity_type,
                                   )
                               )
                     if resolved_entities_list:
                          resolved_entities_output = resolved_entities_list

                answer_value = transformed_value # Update answer_value with resolved entities

    # Determine result chunks - Back to original logic: return first 10 retrieved chunks
    if answer_value in ("not found", None) and query_type != "decomposition":
        result_chunks = []
    else:
        result_chunks = chunks # Remove the [:10] limit

    # Return QueryResult including citations (List[int]), reasoning, and all retrieved chunks
    return QueryResult(
        answer=answer_value,
        chunks=result_chunks,
        citations=citations,
        resolved_entities=resolved_entities_output,
        reasoning=reasoning, # Add reasoning here
        # Pass all_responses from llm_output
        all_responses=llm_output.get("all_responses")
    )


# Convenience functions for specific query types
async def decomposition_query(
    query: str,
    document_id: str,
    rules: List[Rule],
    format: FormatType,
    llm_service: CompletionService,
    vector_db_service: Any,
) -> QueryResult:
    """Process the query based on the decomposition type."""
    return await process_query(
        "decomposition",
        query,
        document_id,
        rules,
        format,
        llm_service,
        vector_db_service,
    )


async def hybrid_query(
    query: str,
    document_id: str,
    rules: List[Rule],
    format: FormatType,
    llm_service: CompletionService,
    vector_db_service: Any,
) -> QueryResult:
    """Process the query based on the hybrid type."""
    return await process_query(
        "hybrid",
        query,
        document_id,
        rules,
        format,
        llm_service,
        vector_db_service,
    )


async def simple_vector_query(
    query: str,
    document_id: str,
    rules: List[Rule],
    format: FormatType,
    llm_service: CompletionService,
    vector_db_service: Any,
) -> QueryResult:
    """Process the query based on the simple vector type."""
    return await process_query(
        "simple_vector",
        query,
        document_id,
        rules,
        format,
        llm_service,
        vector_db_service,
    )


async def inference_query(
    query: str,
    rules: List[Rule],
    format: FormatType,
    llm_service: CompletionService,
) -> QueryResult:
    """Generate a response, no need for vector retrieval."""
    # Since we are just answering this query based on data provided in the query,
    # ther is no need to retrieve any chunks from the vector database.

    answer = await generate_inferred_response(
        llm_service, query, rules, format
    )
    answer_value = answer["answer"]

    # Extract and apply keyword replacements from all resolve_entity rules
    resolve_entity_rules = [
        rule for rule in rules if rule.type == "resolve_entity"
    ]

    if resolve_entity_rules and answer_value:
        # Combine all replacements from all resolve_entity rules
        replacements = {}
        for rule in resolve_entity_rules:
            if rule.options:
                rule_replacements = dict(
                    option.split(":") for option in rule.options
                )
                replacements.update(rule_replacements)

        if replacements:
            print(f"Resolving entities in answer: {answer_value}")
            answer_value = replace_keywords(answer_value, replacements)

    return QueryResult(answer=answer_value, chunks=[])
