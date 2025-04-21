"""Document service."""

import logging
import os
import tempfile
import uuid
from typing import Dict, List, Optional

from langchain.schema import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import Settings
from app.services.llm.base import CompletionService
from app.services.loaders.factory import LoaderFactory
from app.services.vector_db.base import VectorDBService
from app.services.s3_service import S3Service
from app.services.metadata_service import MetadataService
from app.models.document import DocumentMetadata

logger = logging.getLogger(__name__)


class DocumentService:
    """Document service."""

    def __init__(
        self,
        vector_db_service: VectorDBService,
        llm_service: CompletionService,
        settings: Settings,
        s3_service: Optional[S3Service] = None,
        metadata_service: Optional[MetadataService] = None,
    ):
        """Document service."""
        self.vector_db_service = vector_db_service
        self.llm_service = llm_service
        self.settings = settings
        self.s3_service = s3_service
        self.metadata_service = metadata_service
        self.loader_factory = LoaderFactory()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    async def upload_document(
        self,
        filename: str,
        file_content: bytes,
        content_type: Optional[str] = None,
    ) -> Optional[str]:
        """Upload a document, store in vector DB, S3 (if configured), and metadata DB (if configured)."""
        temp_file_path = None
        document_id = self._generate_document_id()
        s3_key = None
        s3_upload_successful = False
        metadata_saved = False

        try:
            logger.info(
                f"Starting upload process for document ID: {document_id}, filename: {filename}"
            )

            # 1. Save the file to a temporary location
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(filename)[1]
            ) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            logger.debug(f"File saved to temporary path: {temp_file_path}")

            # 2. Process the document (load and split)
            chunks = await self._process_document(temp_file_path)
            if not chunks:
                raise ValueError("Document processing resulted in no chunks.")

            # 3. Prepare and upsert chunks to Vector DB
            prepared_chunks = await self.vector_db_service.prepare_chunks(
                document_id, chunks
            )
            await self.vector_db_service.upsert_vectors(prepared_chunks)
            logger.info(
                f"Successfully upserted {len(prepared_chunks)} chunks to vector DB for {document_id}"
            )

            # 4. Upload original file to S3 (if configured)
            if self.s3_service and self.s3_service._is_configured():
                s3_key = self.s3_service.get_object_key(document_id, filename)
                logger.info(f"Attempting to upload original file to S3: {s3_key}")
                s3_upload_successful = self.s3_service.upload_file(
                    file_content=file_content,
                    s3_key=s3_key,
                    content_type=content_type,
                )
                if not s3_upload_successful:
                    # Decide on error handling: For now, log and continue, but don't save metadata
                    logger.error(f"S3 upload failed for {document_id}. Proceeding without S3 storage.")
                    s3_key = None # Ensure s3_key is None if upload failed
                else:
                    logger.info(f"Successfully uploaded original file to S3: {s3_key}")

            # 5. Save metadata to MongoDB (if configured)
            if self.metadata_service and self.metadata_service._is_configured():
                logger.info(f"Attempting to save metadata for {document_id}")
                metadata = DocumentMetadata(
                    id=document_id,
                    filename=filename,
                    s3_key=s3_key,  # Will be None if S3 not configured or upload failed
                    content_type=content_type,
                )
                metadata_saved = self.metadata_service.add_document_metadata(metadata)
                if not metadata_saved:
                    # Decide on error handling: Critical? Log and continue?
                    logger.error(f"Failed to save metadata for {document_id}. Upload may be incomplete.")
                    # Potentially raise an exception here if metadata is critical
                else:
                    logger.info(f"Successfully saved metadata for {document_id}")

            # 6. Return document ID if vector DB upsert was successful
            return document_id

        except Exception as e:
            logger.error(f"Error during upload for {document_id}: {e}", exc_info=True)
            # Basic Rollback attempt (best effort)
            if document_id:
                logger.warning(f"Attempting rollback for failed upload: {document_id}")
                if self.vector_db_service:
                    try:
                        await self.vector_db_service.delete_document(document_id)
                        logger.info(f"Rollback: Deleted vectors for {document_id}")
                    except Exception as rb_e:
                        logger.error(f"Rollback Error (Vector DB): {rb_e}", exc_info=True)

                if s3_upload_successful and s3_key and self.s3_service:
                    try:
                        self.s3_service.delete_file(s3_key)
                        logger.info(f"Rollback: Deleted S3 object {s3_key}")
                    except Exception as rb_e:
                        logger.error(f"Rollback Error (S3): {rb_e}", exc_info=True)

                if metadata_saved and self.metadata_service:
                     try:
                        self.metadata_service.delete_document_metadata(document_id)
                        logger.info(f"Rollback: Deleted metadata for {document_id}")
                     except Exception as rb_e:
                         logger.error(f"Rollback Error (Metadata): {rb_e}", exc_info=True)
            return None
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"Removed temporary file: {temp_file_path}")
                except OSError as e:
                    logger.error(f"Error removing temporary file {temp_file_path}: {e}")

    async def _process_document(
        self, file_path: str
    ) -> List[LangchainDocument]:
        """Process a document."""
        # Load the document
        docs = await self._load_document(file_path)

        # Split the document into chunks
        chunks = self.splitter.split_documents(docs)
        logger.info(f"Document split into {len(chunks)} chunks")
        return chunks

    async def _load_document(self, file_path: str) -> List[LangchainDocument]:

        # Create a loader
        loader = self.loader_factory.create_loader(self.settings)

        if loader is None:
            raise ValueError(
                f"No loader available for configured loader type: {self.settings.loader}"
            )

        # Load the document
        try:
            return await loader.load(file_path)
        except Exception as e:
            logger.error(f"Loader failed: {e}. Unable to load document.")
            raise

    @staticmethod
    def _generate_document_id() -> str:
        return uuid.uuid4().hex

    async def delete_document(self, document_id: str) -> Dict[str, str]:
        """Delete a document from vector DB, S3 (if configured), and metadata DB (if configured)."""
        s3_deleted = False
        metadata_deleted = False
        vector_db_deleted = False
        error_occurred = False
        s3_key = None

        try:
            # 1. Delete from S3 (if configured)
            if self.metadata_service and self.s3_service and self.s3_service._is_configured():
                s3_key = self.metadata_service.get_s3_key_for_document(document_id)
                if s3_key:
                    logger.info(f"Attempting to delete S3 object: {s3_key} for document {document_id}")
                    s3_deleted = self.s3_service.delete_file(s3_key)
                    if not s3_deleted:
                        logger.error(f"Failed to delete S3 object {s3_key} for document {document_id}")
                        error_occurred = True # Log error but continue deletion
                else:
                    logger.info(f"No S3 key found in metadata for document {document_id}, skipping S3 delete.")
                    s3_deleted = True # No key means effectively deleted/nothing to delete
            else:
                 logger.info(f"S3 or Metadata service not configured, skipping S3 delete for {document_id}.")
                 s3_deleted = True # Treat as success if not configured

            # 2. Delete from Metadata DB (if configured)
            if self.metadata_service and self.metadata_service._is_configured():
                logger.info(f"Attempting to delete metadata for document {document_id}")
                metadata_deleted = self.metadata_service.delete_document_metadata(document_id)
                if not metadata_deleted:
                     logger.error(f"Failed to delete metadata for document {document_id}")
                     error_occurred = True # Log error but continue deletion
            else:
                logger.info(f"Metadata service not configured, skipping metadata delete for {document_id}.")
                metadata_deleted = True # Treat as success if not configured

            # 3. Delete from Vector DB (always attempt)
            logger.info(f"Attempting to delete vectors for document {document_id}")
            result = await self.vector_db_service.delete_document(document_id)
            vector_db_deleted = result.get("status") == "success"
            if not vector_db_deleted:
                logger.error(f"Failed to delete vectors for document {document_id}: {result.get('message')}")
                error_occurred = True

            # Determine overall status
            if vector_db_deleted and s3_deleted and metadata_deleted:
                final_status = "success"
                final_message = "Document deleted successfully from all stores."
            else:
                final_status = "partial_failure" if vector_db_deleted else "failure"
                messages = []
                if not vector_db_deleted: messages.append(f"Vector DB delete failed: {result.get('message', 'Unknown error')}")
                if not s3_deleted: messages.append(f"S3 delete failed for key {s3_key}")
                if not metadata_deleted: messages.append("Metadata delete failed")
                final_message = "Deletion incomplete. Failures: " + "; ".join(messages)
                # Raise an exception for partial/full failure to signal upstream?
                # Or just return the status?
                # Let's return the status for now.

            return {"id": document_id, "status": final_status, "message": final_message}

        except Exception as e:
            logger.error(f"Unexpected error deleting document {document_id}: {e}", exc_info=True)
            # Re-raise the exception for the endpoint to handle
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred during deletion of document {document_id}"
            ) from e
