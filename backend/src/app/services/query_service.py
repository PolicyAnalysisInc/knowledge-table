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

    search_response = await search_method(query, document_id, rules)
    retrieved_chunks: List[Chunk] = (
        search_response["chunks"]
        if isinstance(search_response, dict)
        else search_response.chunks
    )

    llm_result = await generate_response(
        llm_service, query, retrieved_chunks, rules, format
    )
    answer_value = llm_result["answer"]
    cited_indices = llm_result["cited_chunk_indices"]

    logger.info(
        f"Query processed. Answer: {answer_value}, Cited Indices: {cited_indices}, Retrieved Chunks: {len(retrieved_chunks)}"
    )

    transformations: Dict[str, Union[str, List[str]]] = {
        "original": "",
        "resolved": "",
    }

    if answer_value is not None and format in ["str", "str_array"]:
        resolve_entity_rules = [
            rule for rule in rules if rule.type == "resolve_entity"
        ]
        replacements: Dict[str, str] = {}
        if resolve_entity_rules:
            for rule in resolve_entity_rules:
                if rule.options:
                    try:
                        rule_replacements = dict(
                            option.split(":", 1) for option in rule.options
                        )
                        replacements.update(rule_replacements)
                    except ValueError as e:
                         logger.warning(f"Skipping invalid resolve_entity rule option in rule {rule}: {e}")

            if replacements:
                print(f"Resolving entities in answer: {answer_value}")
                transformed_value: Union[str, List[str]]
                transform_dict: Dict[str, Union[str, List[str]]]
                if isinstance(answer_value, list):
                    transformed_value, transform_dict = replace_keywords(
                        answer_value, replacements
                    )
                else:
                    transformed_value, transform_dict = replace_keywords(
                         str(answer_value) if not isinstance(answer_value, str) else answer_value,
                         replacements
                    )
                transformations = transform_dict
                answer_value = transformed_value

    result_chunks_to_return = retrieved_chunks[:10]

    return QueryResult(
        answer=answer_value,
        chunks=result_chunks_to_return,
        cited_chunk_indices=cited_indices,
        resolved_entities=(
            [
                ResolvedEntitySchema(
                    original=transformations["original"],
                    resolved=transformations["resolved"],
                    source={"type": "column", "id": "unknown-rule-source"},
                    entityType="unknown-entity-type",
                )
            ]
            if transformations["original"] or transformations["resolved"]
            else None
        ),
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
    logger.info("Processing inference query.")
    answer_dict = await generate_inferred_response(
        llm_service, query, rules, format
    )
    answer_value = answer_dict["answer"]

    transformations: Dict[str, Union[str, List[str]]] = {
        "original": "",
        "resolved": "",
    }
    if answer_value is not None and format in ["str", "str_array"]:
        resolve_entity_rules = [
            rule for rule in rules if rule.type == "resolve_entity"
        ]
        replacements = {}
        if resolve_entity_rules:
            for rule in resolve_entity_rules:
                if rule.options:
                    try:
                        rule_replacements = dict(
                            option.split(":", 1) for option in rule.options
                        )
                        replacements.update(rule_replacements)
                    except ValueError as e:
                        logger.warning(f"Skipping invalid resolve_entity rule option in rule {rule}: {e}")

        if replacements:
            print(f"Resolving entities in inferred answer: {answer_value}")
            transformed_value: Union[str, List[str]]
            transform_dict: Dict[str, Union[str, List[str]]]
            if isinstance(answer_value, list):
                transformed_value, transform_dict = replace_keywords(
                    answer_value, replacements
                )
            else:
                transformed_value, transform_dict = replace_keywords(
                     str(answer_value) if not isinstance(answer_value, str) else answer_value,
                     replacements
                 )
            transformations = transform_dict
            answer_value = transformed_value

    return QueryResult(
        answer=answer_value,
        chunks=[],
        cited_chunk_indices=None,
        resolved_entities=(
             [
                 ResolvedEntitySchema(
                     original=transformations["original"],
                     resolved=transformations["resolved"],
                     source={"type": "column", "id": "unknown-rule-source"},
                     entityType="unknown-entity-type",
                 )
             ]
             if transformations["original"] or transformations["resolved"]
             else None
         )
    )
