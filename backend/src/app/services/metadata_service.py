"""Service for managing document metadata in MongoDB."""

import logging
from typing import Any, Dict, Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, OperationFailure

from app.core.config import Settings
from app.models.document import DocumentMetadata

logger = logging.getLogger(__name__)


class MetadataService:
    """Handles CRUD operations for document metadata in MongoDB."""

    def __init__(self, settings: Settings):
        """Initializes the MongoDB client and collection."""
        self.settings = settings
        self.client: Optional[MongoClient] = None
        self.db = None
        self.collection: Optional[Collection] = None

        if settings.mongodb_uri:
            try:
                self.client = MongoClient(settings.mongodb_uri)
                # The ismaster command is cheap and does not require auth.
                self.client.admin.command("ismaster")
                self.db = self.client[settings.mongodb_database]
                self.collection = self.db["document_metadata"]
                logger.info(
                    f"MongoDB client initialized for database: {settings.mongodb_database}"
                )
            except ConnectionFailure as e:
                logger.error(f"Could not connect to MongoDB: {e}")
                self.client = None
                self.db = None
                self.collection = None
            except Exception as e:
                logger.error(f"An unexpected error occurred connecting to MongoDB: {e}")
                self.client = None
                self.db = None
                self.collection = None
        else:
            logger.warning("MetadataService not initialized. Missing MongoDB URI.")

    def _is_configured(self) -> bool:
        """Check if MongoDB service is properly configured."""
        return self.collection is not None

    def add_document_metadata(self, metadata: DocumentMetadata) -> bool:
        """
        Adds or updates document metadata in MongoDB.

        Args:
            metadata: The DocumentMetadata object to store.

        Returns:
            True if the operation was successful, False otherwise.
        """
        if not self._is_configured():
            logger.warning("Add metadata skipped: MongoDB service not configured.")
            return False

        try:
            # Use model_dump with by_alias=True to get _id correctly
            metadata_dict = metadata.model_dump(by_alias=True)
            self.collection.update_one(
                {"_id": metadata.id},
                {"$set": metadata_dict},
                upsert=True,
            )
            logger.info(f"Successfully added/updated metadata for document ID: {metadata.id}")
            return True
        except OperationFailure as e:
            logger.error(
                f"Failed to add/update metadata for document ID {metadata.id}: {e}"
            )
            return False
        except Exception as e:
            logger.error(
                f"An unexpected error occurred adding/updating metadata for {metadata.id}: {e}"
            )
            return False

    def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """
        Retrieves document metadata from MongoDB.

        Args:
            document_id: The ID of the document.

        Returns:
            A DocumentMetadata object if found, otherwise None.
        """
        if not self._is_configured():
            logger.warning(
                "Get metadata skipped: MongoDB service not configured."
            )
            return None

        try:
            data = self.collection.find_one({"_id": document_id})
            if data:
                return DocumentMetadata(**data)
            return None
        except OperationFailure as e:
            logger.error(f"Failed to retrieve metadata for document ID {document_id}: {e}")
            return None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred retrieving metadata for {document_id}: {e}"
            )
            return None

    def get_s3_key_for_document(self, document_id: str) -> Optional[str]:
        """
        Retrieves the S3 key for a specific document.

        Args:
            document_id: The ID of the document.

        Returns:
            The S3 key if found, otherwise None.
        """
        metadata = self.get_document_metadata(document_id)
        return metadata.s3_key if metadata else None

    def delete_document_metadata(self, document_id: str) -> bool:
        """
        Deletes document metadata from MongoDB.

        Args:
            document_id: The ID of the document to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        if not self._is_configured():
            logger.warning(
                "Delete metadata skipped: MongoDB service not configured."
            )
            # If not configured, arguably nothing to delete, so return True?
            # Or False to indicate action wasn't performed?
            # Let's return False to indicate the service is unavailable.
            return False

        try:
            result = self.collection.delete_one({"_id": document_id})
            if result.deleted_count > 0:
                logger.info(f"Successfully deleted metadata for document ID: {document_id}")
                return True
            else:
                logger.warning(
                    f"Metadata delete requested for {document_id}, but no matching document found."
                )
                # Still considered successful from the service perspective
                return True
        except OperationFailure as e:
            logger.error(f"Failed to delete metadata for document ID {document_id}: {e}")
            return False
        except Exception as e:
            logger.error(
                f"An unexpected error occurred deleting metadata for {document_id}: {e}"
            )
            return False

    def close_connection(self):
        """Closes the MongoDB client connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.") 