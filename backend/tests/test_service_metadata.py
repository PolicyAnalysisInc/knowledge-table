"""Tests for MetadataService."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from pymongo.results import DeleteResult

from app.core.config import Settings
from app.models.document import DocumentMetadata
from app.services.metadata_service import MetadataService

# --- Fixtures --- #

@pytest.fixture
def mock_settings_mongo_configured():
    return Settings(
        mongodb_uri="mongodb://test-mongo:27017/",
        mongodb_database="test_db",
    )

@pytest.fixture
def mock_settings_mongo_not_configured():
    return Settings() # No MongoDB URI

@pytest.fixture
def mock_mongo_client():
    with patch("pymongo.MongoClient") as mock_constructor:
        mock_client_instance = MagicMock(spec=MongoClient)
        mock_db = MagicMock()
        mock_collection = MagicMock()

        # Mock the connection check
        mock_client_instance.admin.command.return_value = {"ok": 1}

        mock_client_instance.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection

        mock_constructor.return_value = mock_client_instance
        yield {
            "client": mock_client_instance,
            "db": mock_db,
            "collection": mock_collection,
        }

# --- Initialization Tests --- #

def test_metadata_service_initialization_configured(mock_settings_mongo_configured, mock_mongo_client):
    """Test MetadataService initializes correctly when configured."""
    service = MetadataService(mock_settings_mongo_configured)
    assert service.client == mock_mongo_client["client"]
    assert service.db == mock_mongo_client["db"]
    assert service.collection == mock_mongo_client["collection"]
    assert service._is_configured() is True
    mock_mongo_client["client"].admin.command.assert_called_once_with("ismaster")
    mock_mongo_client["client"].__getitem__.assert_called_once_with("test_db")
    mock_mongo_client["db"].__getitem__.assert_called_once_with("document_metadata")

def test_metadata_service_initialization_not_configured(mock_settings_mongo_not_configured):
    """Test MetadataService initializes correctly when not configured."""
    service = MetadataService(mock_settings_mongo_not_configured)
    assert service.client is None
    assert service.db is None
    assert service.collection is None
    assert service._is_configured() is False

def test_metadata_service_initialization_connection_failure(mock_settings_mongo_configured):
    """Test initialization handles MongoDB ConnectionFailure."""
    with patch("pymongo.MongoClient") as mock_constructor:
        mock_client_instance = MagicMock()
        mock_client_instance.admin.command.side_effect = ConnectionFailure("Cannot connect")
        mock_constructor.return_value = mock_client_instance

        service = MetadataService(mock_settings_mongo_configured)
        assert service.client is None
        assert service.db is None
        assert service.collection is None
        assert service._is_configured() is False

# --- add_document_metadata Tests --- #

def test_add_document_metadata_success(mock_settings_mongo_configured, mock_mongo_client):
    """Test successfully adding document metadata."""
    service = MetadataService(mock_settings_mongo_configured)
    mock_collection = mock_mongo_client["collection"]
    metadata = DocumentMetadata(
        id="doc1", filename="file1.pdf", s3_key="s3://bucket/key1"
    )

    result = service.add_document_metadata(metadata)

    assert result is True
    mock_collection.update_one.assert_called_once()
    call_args, call_kwargs = mock_collection.update_one.call_args
    assert call_args[0] == {"_id": "doc1"}
    assert "$set" in call_args[1]
    assert call_args[1]["$set"]["_id"] == "doc1"
    assert call_args[1]["$set"]["filename"] == "file1.pdf"
    assert call_args[1]["$set"]["s3_key"] == "s3://bucket/key1"
    assert call_kwargs["upsert"] is True

def test_add_document_metadata_not_configured(mock_settings_mongo_not_configured, mock_mongo_client):
    """Test add metadata is skipped if service not configured."""
    service = MetadataService(mock_settings_mongo_not_configured)
    metadata = DocumentMetadata(id="doc1", filename="file1.pdf")
    result = service.add_document_metadata(metadata)
    assert result is False
    mock_mongo_client["collection"].update_one.assert_not_called()

def test_add_document_metadata_operation_failure(mock_settings_mongo_configured, mock_mongo_client):
    """Test handling OperationFailure during metadata add."""
    service = MetadataService(mock_settings_mongo_configured)
    mock_collection = mock_mongo_client["collection"]
    mock_collection.update_one.side_effect = OperationFailure("DB error")
    metadata = DocumentMetadata(id="doc1", filename="file1.pdf")

    result = service.add_document_metadata(metadata)
    assert result is False
    mock_collection.update_one.assert_called_once()

# --- get_document_metadata Tests --- #

def test_get_document_metadata_success(mock_settings_mongo_configured, mock_mongo_client):
    """Test successfully retrieving document metadata."""
    service = MetadataService(mock_settings_mongo_configured)
    mock_collection = mock_mongo_client["collection"]
    now = datetime.utcnow()
    db_data = {
        "_id": "doc1",
        "filename": "file1.pdf",
        "s3_key": "s3://key",
        "upload_timestamp": now,
        "content_type": "application/pdf"
    }
    mock_collection.find_one.return_value = db_data

    result = service.get_document_metadata("doc1")

    assert result is not None
    assert isinstance(result, DocumentMetadata)
    assert result.id == "doc1"
    assert result.filename == "file1.pdf"
    assert result.s3_key == "s3://key"
    assert result.upload_timestamp == now
    assert result.content_type == "application/pdf"
    mock_collection.find_one.assert_called_once_with({"_id": "doc1"})

def test_get_document_metadata_not_found(mock_settings_mongo_configured, mock_mongo_client):
    """Test retrieving metadata when document ID doesn't exist."""
    service = MetadataService(mock_settings_mongo_configured)
    mock_collection = mock_mongo_client["collection"]
    mock_collection.find_one.return_value = None

    result = service.get_document_metadata("doc_not_found")

    assert result is None
    mock_collection.find_one.assert_called_once_with({"_id": "doc_not_found"})

def test_get_document_metadata_not_configured(mock_settings_mongo_not_configured, mock_mongo_client):
    """Test get metadata is skipped if service not configured."""
    service = MetadataService(mock_settings_mongo_not_configured)
    result = service.get_document_metadata("doc1")
    assert result is None
    mock_mongo_client["collection"].find_one.assert_not_called()

def test_get_document_metadata_operation_failure(mock_settings_mongo_configured, mock_mongo_client):
    """Test handling OperationFailure during metadata get."""
    service = MetadataService(mock_settings_mongo_configured)
    mock_collection = mock_mongo_client["collection"]
    mock_collection.find_one.side_effect = OperationFailure("DB error")

    result = service.get_document_metadata("doc1")
    assert result is None
    mock_collection.find_one.assert_called_once()

# --- get_s3_key_for_document Tests --- #

def test_get_s3_key_for_document_success(mock_settings_mongo_configured, mock_mongo_client):
    """Test retrieving only the S3 key successfully."""
    service = MetadataService(mock_settings_mongo_configured)
    mock_collection = mock_mongo_client["collection"]
    db_data = {"_id": "doc1", "filename": "f.txt", "s3_key": "the_s3_key"}
    mock_collection.find_one.return_value = db_data

    s3_key = service.get_s3_key_for_document("doc1")
    assert s3_key == "the_s3_key"
    mock_collection.find_one.assert_called_once_with({"_id": "doc1"})

def test_get_s3_key_for_document_no_s3_key(mock_settings_mongo_configured, mock_mongo_client):
    """Test retrieving S3 key when it's not set in metadata."""
    service = MetadataService(mock_settings_mongo_configured)
    mock_collection = mock_mongo_client["collection"]
    db_data = {"_id": "doc2", "filename": "f.txt", "s3_key": None} # No s3_key
    mock_collection.find_one.return_value = db_data

    s3_key = service.get_s3_key_for_document("doc2")
    assert s3_key is None
    mock_collection.find_one.assert_called_once_with({"_id": "doc2"})

def test_get_s3_key_for_document_not_found(mock_settings_mongo_configured, mock_mongo_client):
    """Test retrieving S3 key when document metadata is not found."""
    service = MetadataService(mock_settings_mongo_configured)
    mock_collection = mock_mongo_client["collection"]
    mock_collection.find_one.return_value = None

    s3_key = service.get_s3_key_for_document("doc_not_found")
    assert s3_key is None
    mock_collection.find_one.assert_called_once_with({"_id": "doc_not_found"})

# --- delete_document_metadata Tests --- #

def test_delete_document_metadata_success(mock_settings_mongo_configured, mock_mongo_client):
    """Test successfully deleting document metadata."""
    service = MetadataService(mock_settings_mongo_configured)
    mock_collection = mock_mongo_client["collection"]
    mock_collection.delete_one.return_value = DeleteResult({"n": 1}, acknowledged=True)

    result = service.delete_document_metadata("doc1")

    assert result is True
    mock_collection.delete_one.assert_called_once_with({"_id": "doc1"})

def test_delete_document_metadata_not_found(mock_settings_mongo_configured, mock_mongo_client):
    """Test deleting metadata when document ID doesn't exist (should still return True)."""
    service = MetadataService(mock_settings_mongo_configured)
    mock_collection = mock_mongo_client["collection"]
    mock_collection.delete_one.return_value = DeleteResult({"n": 0}, acknowledged=True)

    result = service.delete_document_metadata("doc_not_found")

    assert result is True # Considered success as the state is achieved
    mock_collection.delete_one.assert_called_once_with({"_id": "doc_not_found"})

def test_delete_document_metadata_not_configured(mock_settings_mongo_not_configured, mock_mongo_client):
    """Test delete metadata is skipped if service not configured (returns False)."""
    service = MetadataService(mock_settings_mongo_not_configured)
    result = service.delete_document_metadata("doc1")
    assert result is False
    mock_mongo_client["collection"].delete_one.assert_not_called()

def test_delete_document_metadata_operation_failure(mock_settings_mongo_configured, mock_mongo_client):
    """Test handling OperationFailure during metadata delete."""
    service = MetadataService(mock_settings_mongo_configured)
    mock_collection = mock_mongo_client["collection"]
    mock_collection.delete_one.side_effect = OperationFailure("DB error")

    result = service.delete_document_metadata("doc1")
    assert result is False
    mock_collection.delete_one.assert_called_once()

# --- close_connection Tests --- #

def test_close_connection(mock_settings_mongo_configured, mock_mongo_client):
    """Test closing the MongoDB connection."""
    service = MetadataService(mock_settings_mongo_configured)
    mock_client = mock_mongo_client["client"]

    service.close_connection()
    mock_client.close.assert_called_once()

def test_close_connection_not_configured(mock_settings_mongo_not_configured):
    """Test closing connection does nothing if not configured."""
    # This primarily tests that it doesn't raise an error
    service = MetadataService(mock_settings_mongo_not_configured)
    try:
        service.close_connection()
    except Exception as e:
        pytest.fail(f"close_connection raised an unexpected exception: {e}") 