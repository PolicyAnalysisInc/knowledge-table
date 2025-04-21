"""Dependencies for the application."""

from fastapi import Depends
from typing import Optional

from app.core.config import Settings, get_settings
from app.services.document_service import DocumentService
from app.services.embedding.base import EmbeddingService
from app.services.embedding.factory import EmbeddingServiceFactory
from app.services.llm.base import CompletionService
from app.services.llm.factory import CompletionServiceFactory
from app.services.vector_db.base import VectorDBService
from app.services.vector_db.factory import VectorDBFactory
from app.services.s3_service import S3Service
from app.services.metadata_service import MetadataService

# Singletons for services that manage external connections
_s3_service_instance: Optional[S3Service] = None
_metadata_service_instance: Optional[MetadataService] = None


def get_llm_service(
    settings: Settings = Depends(get_settings),
) -> CompletionService:
    """Get the LLM service for the application."""
    llm_service = CompletionServiceFactory.create_service(settings)
    if llm_service is None:
        raise ValueError(
            f"Failed to create LLM service for provider: {settings.llm_provider}"
        )
    return llm_service


def get_embedding_service(
    settings: Settings = Depends(get_settings),
) -> EmbeddingService:
    """Get the embedding service for the application."""
    embedding_service = EmbeddingServiceFactory.create_service(settings)
    if embedding_service is None:
        raise ValueError(
            f"Failed to create embedding service for provider: {settings.embedding_provider}"
        )
    return embedding_service


def get_vector_db_service(
    settings: Settings = Depends(get_settings),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    llm_service: CompletionService = Depends(get_llm_service),
) -> VectorDBService:
    """Get the vector database service for the application."""
    vector_db_service = VectorDBFactory.create_vector_db_service(
        embedding_service, llm_service, settings
    )
    if vector_db_service is None:
        raise ValueError(
            f"Failed to create vector database service for provider: {settings.vector_db_provider}"
        )
    return vector_db_service


def get_s3_service(settings: Settings = Depends(get_settings)) -> Optional[S3Service]:
    """Get the S3 service singleton."""
    global _s3_service_instance
    if _s3_service_instance is None:
        _s3_service_instance = S3Service(settings)
    return _s3_service_instance


def get_metadata_service(settings: Settings = Depends(get_settings)) -> Optional[MetadataService]:
    """Get the Metadata service singleton."""
    global _metadata_service_instance
    if _metadata_service_instance is None:
        _metadata_service_instance = MetadataService(settings)
    # TODO: Consider adding logic to handle MongoDB connection closing on shutdown
    return _metadata_service_instance


def get_document_service(
    settings: Settings = Depends(get_settings),
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
    llm_service: CompletionService = Depends(get_llm_service),
    s3_service: Optional[S3Service] = Depends(get_s3_service),
    metadata_service: Optional[MetadataService] = Depends(get_metadata_service),
) -> DocumentService:
    """Get the document service for the application."""
    return DocumentService(
        vector_db_service=vector_db_service,
        llm_service=llm_service,
        settings=settings,
        s3_service=s3_service,
        metadata_service=metadata_service,
    )
