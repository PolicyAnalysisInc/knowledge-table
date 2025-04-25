"""The functions for generating responses from the language model."""

import json
import logging
from typing import Any, List, Tuple, Type, Union, Dict, Optional

from app.models.llm_responses import (
    BaseResponseModel,
    BoolResponseModel,
    IntArrayResponseModel,
    IntResponseModel,
    KeywordsResponseModel,
    NumberArrayResponseModel,
    NumberResponseModel,
    SchemaResponseModel,
    StrArrayResponseModel,
    StrResponseModel,
    SubQueriesResponseModel,
)
from app.models.query_core import FormatType, Rule
from app.models.table import Table
from app.services.llm.base import CompletionService
from app.services.llm.openai_prompts import (
    BASE_PROMPT,
    BOOL_INSTRUCTIONS,
    DECOMPOSE_QUERY_PROMPT,
    INFERRED_BASE_PROMPT,
    NUMBER_ARRAY_INSTRUCTIONS,
    KEYWORD_PROMPT,
    SCHEMA_PROMPT,
    SIMILAR_KEYWORDS_PROMPT,
    STR_ARRAY_INSTRUCTIONS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_model_and_instructions(
    format: str, rules: list[Rule], query: str
) -> Tuple[Type[BaseResponseModel], str]:
    """
    Get the appropriate output model and instructions based on the format.

    Parameters
    ----------
    format : str
        The desired format of the response.
    rules : list[Rule]
        A list of rules to apply when generating the response.
    query : str
        The user's query to be answered.

    Returns
    -------
    Tuple[Type[BaseResponseModel], str]
        A tuple containing the appropriate output model and format-specific instructions.
    """
    str_rule = next(
        (rule for rule in rules if rule.type in ["must_return", "may_return"]),
        None,
    )
    int_rule = next(
        (rule for rule in rules if rule.type == "max_length"), None
    )

    if format == "bool":
        return BoolResponseModel, BOOL_INSTRUCTIONS
    elif format in ["str_array", "str"]:
        str_rule_line = _get_str_rule_line(str_rule, query)
        int_rule_line = _get_int_rule_line(int_rule)
        instructions = STR_ARRAY_INSTRUCTIONS.substitute(
            str_rule_line=str_rule_line, int_rule_line=int_rule_line
        )
        return (
            StrArrayResponseModel
            if format == "str_array"
            else StrResponseModel
        ), instructions
    elif format in ["int", "int_array"]:
        int_rule_line = _get_int_rule_line(int_rule)
        instructions = NUMBER_ARRAY_INSTRUCTIONS.substitute(
            int_rule_line=int_rule_line
        )
        return (
            IntArrayResponseModel
            if format == "int_array"
            else IntResponseModel
        ), instructions
    elif format in ["number", "number_array"]:
        int_rule_line = _get_int_rule_line(int_rule)
        instructions = NUMBER_ARRAY_INSTRUCTIONS.substitute(
            int_rule_line=int_rule_line
        )
        return (
            NumberArrayResponseModel
            if format == "number_array"
            else NumberResponseModel
        ), instructions
    else:
        raise ValueError(f"Unsupported format: {format}")


async def generate_response(
    llm_service: CompletionService,
    query: str,
    chunks: str,
    rules: list[Rule],
    format: FormatType,
) -> Dict[str, Any]:
    """
    Generate a response from the language model based on the given query and format.

    Parameters
    ----------
    llm_service : CompletionService
        The language model service to use for generating the response.
    query : str
        The user's query to be answered.
    chunks : str
        The context or relevant text chunks for answering the query.
    rules : list[Rule]
        A list of rules to apply when generating the response.
    format : FormatType
        The desired format of the response.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the generated answer and confidence or None if an error occurs.
    """
    logger.info(f"Generating response for query: {query} in format: {format}")

    output_model, format_specific_instructions = _get_model_and_instructions(
        format, rules, query
    )

    prompt = BASE_PROMPT.substitute(
        query=query,
        chunks=chunks,
        format_specific_instructions=format_specific_instructions,
    )

    try:
        response: Optional[BaseResponseModel] = await llm_service.generate_completion(prompt, output_model)
        logger.info(f"Raw response from LLM: {response}")

        if response is None:
            logger.warning("LLM returned None object")
            return {"answer": None, "confidence": None, "reasoning": None, "citations": [], "all_responses": None}

        # Extract answer, confidence, reasoning, and citations
        answer = getattr(response, 'answer', None)
        # Handle non-answer models
        if answer is None:
            if isinstance(response, KeywordsResponseModel):
                answer = getattr(response, 'keywords', None)
            elif isinstance(response, SubQueriesResponseModel):
                answer = getattr(response, 'sub_queries', None)
            elif isinstance(response, SchemaResponseModel):
                 answer = getattr(response, 'relationships', None)

        confidence = getattr(response, 'confidence', None)
        reasoning = getattr(response, 'reasoning', None)
        # Citations should now always exist due to model validation, default to [] if somehow None
        citations = getattr(response, 'citations', [])

        logger.info(f"Processed response: answer={answer}, confidence={confidence}, reasoning={reasoning}, citations={citations}")
        # Include all_responses in the returned dictionary
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning": reasoning,
            "citations": citations,
            "all_responses": getattr(response, 'all_responses', None)
        }
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        # Ensure consistent return structure on error
        return {"answer": None, "confidence": None, "reasoning": None, "citations": [], "all_responses": None}


async def generate_inferred_response(
    llm_service: CompletionService,
    query: str,
    rules: list[Rule],
    format: FormatType,
) -> Dict[str, Any]:
    """
    Generate a response from the language model based on the given query and format.

    Parameters
    ----------
    llm_service : CompletionService
        The language model service to use for generating the response.
    query : str
        The user's query to be answered.
    rules : list[Rule]
        A list of rules to apply when generating the response.
    format : FormatType
        The desired format of the response.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the generated answer and confidence or None if an error occurs.
    """
    logger.info(
        f"Generating inferred response for query: {query} in format: {format}"
    )

    output_model, format_specific_instructions = _get_model_and_instructions(
        format, rules, query
    )
    prompt = INFERRED_BASE_PROMPT.substitute(
        query=query,
        format_specific_instructions=format_specific_instructions,
    )

    try:
        response: Optional[BaseResponseModel] = await llm_service.generate_completion(prompt, output_model)
        logger.info(f"Raw response from LLM: {response}")

        if response is None:
            logger.warning("LLM returned None object")
            return {"answer": None, "confidence": None, "reasoning": None, "citations": [], "all_responses": None}

        # Extract answer, confidence, reasoning, and citations
        answer = getattr(response, 'answer', None)
        # Handle non-answer models
        if answer is None:
             if isinstance(response, KeywordsResponseModel):
                 answer = getattr(response, 'keywords', None)
             elif isinstance(response, SubQueriesResponseModel):
                 answer = getattr(response, 'sub_queries', None)
             elif isinstance(response, SchemaResponseModel):
                  answer = getattr(response, 'relationships', None)

        confidence = getattr(response, 'confidence', None)
        reasoning = getattr(response, 'reasoning', None)
        # Citations should now always exist due to model validation, default to [] if somehow None
        citations = getattr(response, 'citations', [])

        logger.info(f"Processed inferred response: answer={answer}, confidence={confidence}, reasoning={reasoning}, citations={citations}")
        # Include all_responses in the returned dictionary
        return {
            "answer": answer,
            "confidence": confidence,
            "reasoning": reasoning,
            "citations": citations,
            "all_responses": getattr(response, 'all_responses', None)
        }
    except Exception as e:
        logger.error(f"Error generating inferred response: {str(e)}", exc_info=True)
        # Ensure consistent return structure on error
        return {"answer": None, "confidence": None, "reasoning": None, "citations": [], "all_responses": None}


async def get_keywords(
    llm_service: CompletionService, query: str
) -> Dict[str, Any]:
    """
    Extract keywords from a query using the language model.

    Parameters
    ----------
    llm_service : CompletionService
        The language model service to use for keyword extraction.
    query : str
        The query from which to extract keywords.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the extracted keywords and confidence or None if an error occurs.
    """
    logger.info(f"Extracting keywords for query: {query}")

    try:
        prompt = KEYWORD_PROMPT.substitute(query=query)
        response: Optional[KeywordsResponseModel] = await llm_service.generate_completion(
            prompt, KeywordsResponseModel
        )
        logger.info(f"Raw keyword response from LLM: {response}")

        if response is None:
            logger.warning("Keyword LLM returned None object")
            return {"keywords": None, "confidence": None, "reasoning": None, "citations": []}

        keywords = getattr(response, 'keywords', None)
        confidence = getattr(response, 'confidence', None)
        reasoning = getattr(response, 'reasoning', None)
        # Citations should now always exist due to model validation, default to [] if somehow None
        citations = getattr(response, 'citations', [])

        logger.info(f"Processed keywords: keywords={keywords}, confidence={confidence}, reasoning={reasoning}, citations={citations}")
        # Return structure aligns with BaseResponseModel fields + specific field ('keywords')
        return {"keywords": keywords, "confidence": confidence, "reasoning": reasoning, "citations": citations}
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}", exc_info=True)
        return {"keywords": None, "confidence": None, "reasoning": None, "citations": []}


async def get_similar_keywords(
    llm_service: CompletionService, chunks: str, rule: list[str]
) -> Dict[str, Any]:
    """
    Retrieve keywords similar to the provided keywords from the given text chunks.

    Parameters
    ----------
    llm_service : CompletionService
        The language model service to use for finding similar keywords.
    chunks : str
        The text chunks to search for similar keywords.
    rule : list[str]
        The list of keywords to use as a reference for finding similar ones.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the similar keywords found and confidence or None if an error occurs.
    """
    logger.info(f"Finding similar keywords based on rule: {rule}")

    try:
        rule_str = ", ".join(rule) if isinstance(rule, list) else str(rule)
        prompt = SIMILAR_KEYWORDS_PROMPT.substitute(rule=rule_str, chunks=chunks)
        response: Optional[KeywordsResponseModel] = await llm_service.generate_completion(
            prompt, KeywordsResponseModel
        )
        logger.info(f"Raw similar keywords response from LLM: {response}")

        if response is None:
            logger.warning("Similar keywords LLM returned None object")
            return {"keywords": None, "confidence": None, "reasoning": None, "citations": []}

        keywords = getattr(response, 'keywords', None)
        confidence = getattr(response, 'confidence', None)
        reasoning = getattr(response, 'reasoning', None)
        # Citations should now always exist due to model validation, default to [] if somehow None
        citations = getattr(response, 'citations', [])

        logger.info(f"Processed similar keywords: keywords={keywords}, confidence={confidence}, reasoning={reasoning}, citations={citations}")
        return {"keywords": keywords, "confidence": confidence, "reasoning": reasoning, "citations": citations}
    except Exception as e:
        logger.error(f"Error finding similar keywords: {str(e)}", exc_info=True)
        return {"keywords": None, "confidence": None, "reasoning": None, "citations": []}


async def decompose_query(
    llm_service: CompletionService, query: str
) -> Dict[str, Any]:
    """
    Decompose a complex query into multiple simpler sub-queries.

    Parameters
    ----------
    llm_service : CompletionService
        The language model service to use for query decomposition.
    query : str
        The complex query to be decomposed.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the list of sub-queries and confidence or None if an error occurs.
    """
    logger.info(f"Decomposing query: {query}")

    try:
        prompt = DECOMPOSE_QUERY_PROMPT.substitute(query=query)
        response: Optional[SubQueriesResponseModel] = await llm_service.generate_completion(
            prompt, SubQueriesResponseModel
        )
        logger.info(f"Raw decompose query response from LLM: {response}")

        if response is None:
            logger.warning("Decompose query LLM returned None object")
            return {"sub_queries": None, "confidence": None, "reasoning": None, "citations": []}

        sub_queries = getattr(response, 'sub_queries', None)
        confidence = getattr(response, 'confidence', None)
        reasoning = getattr(response, 'reasoning', None)
        # Citations should now always exist due to model validation, default to [] if somehow None
        citations = getattr(response, 'citations', [])

        logger.info(f"Processed sub-queries: sub_queries={sub_queries}, confidence={confidence}, reasoning={reasoning}, citations={citations}")
        return {"sub_queries": sub_queries, "confidence": confidence, "reasoning": reasoning, "citations": citations}
    except Exception as e:
        logger.error(f"Error decomposing query: {str(e)}", exc_info=True)
        return {"sub_queries": None, "confidence": None, "reasoning": None, "citations": []}


async def generate_schema(
    llm_service: CompletionService, data: Table
) -> Dict[str, Any]:
    """
    Generate a schema for the table based on column information and questions.

    Parameters
    ----------
    llm_service : CompletionService
        The language model service to use for schema generation.
    data : Table
        The table data containing information about columns, rows, and documents.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the generated schema and confidence or None if an error occurs.
    """
    logger.info("Generating schema.")

    try:
        documents: List[str] = list(
            set(str(row.document.name) for row in data.rows if row.document)
        )
        columns_data = []
        entity_types_set = set()

        for column in data.columns:
            if column.prompt:
                 col_info = {
                     "id": column.id,
                     "entity_type": column.prompt.entityType,
                     "type": column.prompt.type,
                     "question": column.prompt.query,
                 }
                 columns_data.append(col_info)
                 entity_types_set.add(column.prompt.entityType)
            else:
                 logger.warning(f"Column with id {column.id} is missing prompt data. Skipping for schema generation.")

        if not columns_data:
             logger.error("No valid column data with prompts found to generate schema.")
             return {"relationships": None, "confidence": None, "reasoning": None, "citations": []}

        entity_types = list(entity_types_set)

        prompt = SCHEMA_PROMPT.substitute(
            documents=", ".join(documents) if documents else "",
            entity_types=", ".join(entity_types) if entity_types else "",
            columns=json.dumps(columns_data),
        )

        response: Optional[SchemaResponseModel] = await llm_service.generate_completion(
            prompt, SchemaResponseModel
        )
        logger.info(f"Raw schema response from LLM: {response}")

        if response is None:
            logger.warning("Generate schema LLM returned None object")
            return {"relationships": None, "confidence": None, "reasoning": None, "citations": []}

        relationships = getattr(response, 'relationships', None)
        confidence = getattr(response, 'confidence', None)
        reasoning = getattr(response, 'reasoning', None)
        # Citations should now always exist due to model validation, default to [] if somehow None
        citations = getattr(response, 'citations', [])

        logger.info(f"Processed schema: relationships={relationships}, confidence={confidence}, reasoning={reasoning}, citations={citations}")
        return {"relationships": relationships, "confidence": confidence, "reasoning": reasoning, "citations": citations}

    except Exception as e:
        logger.error(f"Error generating schema: {str(e)}", exc_info=True)
        return {"relationships": None, "confidence": None, "reasoning": None, "citations": []}


def _get_str_rule_line(str_rule: Rule | None, query: str) -> str:
    """
    Generate a string rule line based on the given string rule and query.

    Parameters
    ----------
    str_rule : Rule | None
        The string rule to process.
    query : str
        The original query for context.

    Returns
    -------
    str
        A formatted string containing instructions based on the rule, or an empty string if no rule is provided.
    """
    if str_rule:
        if str_rule.type == "must_return" and str_rule.options:
            options_str = ", ".join(
                f'"{option}"' for option in str_rule.options
            )
            return f"You should only consider these possible values when answering the question: {options_str}. If these values do not exist in the raw text chunks, or if they do not correctly answer the question, respond with None."
        elif str_rule.type == "may_return" and str_rule.options:
            options_str = ", ".join(str_rule.options)
            return f"For example: Query: {query} Response: {options_str}, etc... If you cannot find a related, correct answer in the raw text chunks, respond with None."
    return ""


def _get_int_rule_line(int_rule: Rule | None) -> str:
    """
    Generate an integer rule line based on the given integer rule.

    Parameters
    ----------
    int_rule : Rule | None
        The integer rule to process.

    Returns
    -------
    str
        A formatted string containing instructions based on the rule, or an empty string if no rule is provided.
    """
    if int_rule and int_rule.length is not None:
        return f"Your answer should only return up to {int_rule.length} items. If you have to choose between multiple, return those that answer the question the best. If you cannot find any suitable answer, respond with None."
    return ""
