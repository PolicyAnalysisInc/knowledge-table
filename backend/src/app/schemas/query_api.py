"""Query schemas for API requests and responses."""

from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from app.models.query_core import Chunk, FormatType, Rule


class ResolvedEntitySchema(BaseModel):
    """Schema for resolved entity transformations."""

    original: Union[str, List[str]]
    resolved: Union[str, List[str]]
    source: dict[str, str]
    entityType: str


class QueryPromptSchema(BaseModel):
    """Schema for the prompt part of the query request."""

    id: str
    entity_type: str
    query: str
    type: FormatType
    rules: list[Rule] = []


class QueryRequestSchema(BaseModel):
    """Query request schema."""

    document_id: str
    prompt: QueryPromptSchema

    model_config = ConfigDict(extra="allow")


class VectorResponseSchema(BaseModel):
    """Vector response schema."""

    message: str
    chunks: List[Chunk]
    keywords: Optional[List[str]] = None


class QueryResult(BaseModel):
    """Query result schema returned by internal query services."""

    answer: Any
    chunks: List[Chunk]
    citations: List[int]
    resolved_entities: Optional[List[ResolvedEntitySchema]] = None
    reasoning: Optional[str] = None


class QueryResponseSchema(BaseModel):
    """Query response schema."""

    id: str
    document_id: str
    prompt_id: str
    answer: Optional[Any] = None
    chunks: List[Chunk]
    type: str
    citations: List[str]
    resolved_entities: Optional[List[ResolvedEntitySchema]] = None


class QueryAnswer(BaseModel):
    """Query answer model (part of the final API response)."""

    id: str
    document_id: str
    prompt_id: str
    answer: Optional[Union[int, float, str, bool, List[int], List[float], List[str]]]
    type: str


class QueryAnswerResponse(BaseModel):
    """Query answer response model (the final API response structure)."""

    answer: QueryAnswer
    chunks: List[Chunk]
    citations: List[int]
    resolved_entities: Optional[List[ResolvedEntitySchema]] = None
    reasoning: Optional[str] = None


# Type for search responses (used in service layer)
SearchResponse = Union[dict[str, List[Chunk]], VectorResponseSchema]

# Assuming VectorResponseSchema is defined elsewhere or replacing with appropriate type
# If VectorResponseSchema is not defined, this might need adjustment.
# For now, let's define a placeholder if it's missing:
try:
    from app.schemas.vector_db import VectorResponseSchema # Adjust import path if necessary
except ImportError:
    class VectorResponseSchema(BaseModel): # Placeholder
        chunks: List[Chunk]
