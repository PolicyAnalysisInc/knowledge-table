"""The base class for the vector database services."""

import logging
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from langchain.schema import Document
from pydantic import BaseModel, Field

from app.models.query_core import Rule
from app.schemas.query_api import VectorResponseSchema
from app.services.embedding.base import EmbeddingService
from app.services.llm.base import CompletionService
from app.services.llm_service import get_keywords

logger = logging.getLogger(__name__)


class Metadata(BaseModel, extra="forbid"):
    """Metadata stored in vector storage."""

    text: str
    page_number: int
    chunk_number: int
    document_id: str
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))


class VectorDBService(ABC):
    """The base class for the vector database services."""

    embedding_service: EmbeddingService

    @abstractmethod
    async def upsert_vectors(
        self, vectors: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Upsert the vectors into the vector database."""
        pass

    @abstractmethod
    async def vector_search(
        self, queries: List[str], document_id: str
    ) -> VectorResponseSchema:
        """Perform a vector search."""
        pass

    # Update other methods if they also return VectorResponse
    @abstractmethod
    async def keyword_search(
        self, query: str, document_id: str, keywords: List[str]
    ) -> VectorResponseSchema:
        """Perform a keyword search."""
        pass

    @abstractmethod
    async def hybrid_search(
        self, query: str, document_id: str, rules: List[Rule]
    ) -> VectorResponseSchema:
        """Perform a hybrid search."""
        pass

    @abstractmethod
    async def decomposed_search(
        self, query: str, document_id: str, rules: List[Rule]
    ) -> Dict[str, Any]:
        """Decomposition query."""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> Dict[str, str]:
        """Delete the document from the vector database."""
        pass

    @abstractmethod
    async def ensure_collection_exists(self) -> None:
        """Ensure the collection exists in the vector database."""
        pass

    async def get_embeddings(
        self, texts: Union[str, List[str]]
    ) -> List[List[float]]:
        """Get embeddings for the given text(s) using the embedding service."""
        if isinstance(texts, str):
            texts = [texts]
        return await self.embedding_service.get_embeddings(texts)

    async def get_single_embedding(self, text: str) -> List[float]:
        """Get a single embedding for the given text."""
        embeddings = await self.get_embeddings(text)
        return embeddings[0]

    async def prepare_chunks(
        self, document_id: str, chunks: List[Document]
    ) -> List[Dict[str, Any]]:
        """Prepare chunks for insertion into the vector database."""
        logger.info(f"Preparing {len(chunks)} chunks for document {document_id}")

        # Clean the chunks
        cleaned_texts = [
            re.sub(r"\s+", " ", chunk.page_content.strip()) for chunk in chunks
        ]

        logger.info("Generating embeddings.")

        # Embed all chunks at once
        embedded_chunks = await self.get_embeddings(cleaned_texts)

        # Prepare the data for insertion with detailed logging for page numbers
        prepared_data = []
        for i, (chunk, text, embedding) in enumerate(
            zip(chunks, cleaned_texts, embedded_chunks)
        ):
            metadata_page = chunk.metadata.get("page")
            final_page_number: int

            if metadata_page is not None:
                try:
                    # Attempt to convert metadata page to int
                    page_from_metadata = int(metadata_page)
                    # ADD 1 TO THE METADATA PAGE NUMBER
                    final_page_number = page_from_metadata + 1
                    logger.info(f"Chunk {i}: Found 'page' in metadata: {metadata_page}. Assigning page_number: {final_page_number} (metadata + 1)")
                except (ValueError, TypeError):
                    logger.warning(f"Chunk {i}: Could not convert metadata 'page' ({metadata_page}) to int. Using fallback.")
                    # Fallback logic remains the same (starts from 1)
                    fallback_page_number = i // 5 + 1
                    logger.info(f"Chunk {i}: Using fallback page calculation: {i} // 5 + 1 = {fallback_page_number}")
                    final_page_number = fallback_page_number
            else:
                # Metadata 'page' key not found, use fallback (starts from 1)
                fallback_page_number = i // 5 + 1
                logger.info(f"Chunk {i}: Did not find 'page' in metadata. Using fallback: {i} // 5 + 1 = {fallback_page_number}")
                final_page_number = fallback_page_number

            payload = {
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "text": text,
                "page_number": final_page_number, # Use the potentially incremented page number
                "chunk_number": i,
                "document_id": document_id,
            }
            prepared_data.append(payload)

        return prepared_data

    async def extract_keywords(
        self, query: str, rules: list[Rule], llm_service: CompletionService
    ) -> list[str]:
        """Extract keywords from a user query."""
        keywords = []
        if rules:
            for rule in rules:
                if rule.type in ["must_return", "may_return"]:
                    if rule.options:
                        if isinstance(rule.options, list):
                            keywords.extend(rule.options)
                        elif isinstance(rule.options, dict):
                            for value in rule.options.values():
                                if isinstance(value, list):
                                    keywords.extend(value)
                                elif isinstance(value, str):
                                    keywords.append(value)

        if not keywords:
            extracted_keywords = await get_keywords(llm_service, query)
            if extracted_keywords and isinstance(extracted_keywords, list):
                keywords = extracted_keywords

        return keywords
