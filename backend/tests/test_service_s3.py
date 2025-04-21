"""Tests for S3Service."""

import pytest
from unittest.mock import MagicMock, patch

import boto3
from botocore.exceptions import ClientError

from app.core.config import Settings
from app.services.s3_service import S3Service

# Minimal settings required for S3Service initialization tests
@pytest.fixture
def mock_settings_s3_configured():
    return Settings(
        aws_access_key_id="test_key_id",
        aws_secret_access_key="test_secret_key",
        aws_region="us-east-1",
        s3_bucket_name="test-bucket",
        s3_prefix="test_prefix/",
    )

@pytest.fixture
def mock_settings_s3_not_configured():
    return Settings() # No S3 settings provided

@pytest.fixture
def mock_boto3_client():
    with patch("boto3.client") as mock_client_constructor:
        mock_s3 = MagicMock()
        mock_client_constructor.return_value = mock_s3
        yield mock_s3

# --- Initialization Tests --- #

def test_s3_service_initialization_configured(mock_settings_s3_configured, mock_boto3_client):
    """Test S3Service initializes correctly when configured."""
    service = S3Service(mock_settings_s3_configured)
    assert service.s3_client is not None
    assert service.bucket_name == "test-bucket"
    assert service._is_configured() is True
    mock_boto3_client.assert_called_once_with(
        "s3",
        aws_access_key_id="test_key_id",
        aws_secret_access_key="test_secret_key",
        region_name="us-east-1",
    )

def test_s3_service_initialization_not_configured(mock_settings_s3_not_configured):
    """Test S3Service initializes correctly when not configured."""
    service = S3Service(mock_settings_s3_not_configured)
    assert service.s3_client is None
    assert service.bucket_name is None
    assert service._is_configured() is False

# --- get_object_key Tests --- #

def test_get_object_key(mock_settings_s3_configured):
    """Test constructing the S3 object key."""
    service = S3Service(mock_settings_s3_configured)
    doc_id = "doc123"
    filename = "My Document!.pdf"
    expected_key = "test_prefix/doc123/MyDocument.pdf" # Sanitized filename
    assert service.get_object_key(doc_id, filename) == expected_key

def test_get_object_key_no_prefix(mock_settings_s3_configured):
    """Test constructing the S3 object key when prefix has no trailing slash initially."""
    mock_settings_s3_configured.s3_prefix = "test_prefix" # No trailing slash
    service = S3Service(mock_settings_s3_configured)
    doc_id = "doc456"
    filename = "report_final_v2.txt"
    expected_key = "test_prefix/doc456/report_final_v2.txt"
    assert service.get_object_key(doc_id, filename) == expected_key

# --- upload_file Tests --- #

def test_upload_file_success(mock_settings_s3_configured, mock_boto3_client):
    """Test successful file upload to S3."""
    service = S3Service(mock_settings_s3_configured)
    file_content = b"some test data"
    s3_key = "test_prefix/doc123/test.txt"
    content_type = "text/plain"

    result = service.upload_file(file_content, s3_key, content_type)

    assert result is True
    mock_boto3_client.put_object.assert_called_once_with(
        Bucket="test-bucket", Key=s3_key, Body=file_content, ContentType=content_type
    )

def test_upload_file_success_no_content_type(mock_settings_s3_configured, mock_boto3_client):
    """Test successful file upload without content type."""
    service = S3Service(mock_settings_s3_configured)
    file_content = b"more data"
    s3_key = "test_prefix/doc456/image.jpg"

    result = service.upload_file(file_content, s3_key)

    assert result is True
    mock_boto3_client.put_object.assert_called_once_with(
        Bucket="test-bucket", Key=s3_key, Body=file_content
    )

def test_upload_file_not_configured(mock_settings_s3_not_configured, mock_boto3_client):
    """Test upload is skipped when S3 is not configured."""
    service = S3Service(mock_settings_s3_not_configured)
    result = service.upload_file(b"data", "key", "type")
    assert result is True # Returns True as it didn't block
    mock_boto3_client.put_object.assert_not_called()

def test_upload_file_client_error(mock_settings_s3_configured, mock_boto3_client):
    """Test upload failure due to ClientError."""
    service = S3Service(mock_settings_s3_configured)
    mock_boto3_client.put_object.side_effect = ClientError({'Error': {'Code': 'AccessDenied', 'Message': 'Denied'}}, 'PutObject')

    result = service.upload_file(b"data", "key", "type")
    assert result is False
    mock_boto3_client.put_object.assert_called_once()

def test_upload_file_unexpected_error(mock_settings_s3_configured, mock_boto3_client):
    """Test upload failure due to an unexpected exception."""
    service = S3Service(mock_settings_s3_configured)
    mock_boto3_client.put_object.side_effect = Exception("Something broke")

    result = service.upload_file(b"data", "key", "type")
    assert result is False
    mock_boto3_client.put_object.assert_called_once()

# --- delete_file Tests --- #

def test_delete_file_success(mock_settings_s3_configured, mock_boto3_client):
    """Test successful file deletion from S3."""
    service = S3Service(mock_settings_s3_configured)
    s3_key = "test_prefix/doc123/test.txt"

    result = service.delete_file(s3_key)

    assert result is True
    mock_boto3_client.delete_object.assert_called_once_with(Bucket="test-bucket", Key=s3_key)

def test_delete_file_not_configured(mock_settings_s3_not_configured, mock_boto3_client):
    """Test delete is skipped when S3 is not configured."""
    service = S3Service(mock_settings_s3_not_configured)
    result = service.delete_file("key")
    assert result is True # Returns True as it didn't block
    mock_boto3_client.delete_object.assert_not_called()

def test_delete_file_client_error(mock_settings_s3_configured, mock_boto3_client):
    """Test delete failure due to ClientError (still returns False)."""
    service = S3Service(mock_settings_s3_configured)
    mock_boto3_client.delete_object.side_effect = ClientError({'Error': {'Code': 'SomeError', 'Message': 'Msg'}}, 'DeleteObject')
    s3_key = "test_prefix/doc123/test.txt"

    result = service.delete_file(s3_key)
    assert result is False
    mock_boto3_client.delete_object.assert_called_once()

def test_delete_file_client_error_no_such_key(mock_settings_s3_configured, mock_boto3_client):
    """Test delete handling of NoSuchKey error (still returns False based on current impl)."""
    # Note: Depending on desired behavior, might want this to return True for idempotency.
    # Current impl returns False on any ClientError.
    service = S3Service(mock_settings_s3_configured)
    mock_boto3_client.delete_object.side_effect = ClientError({'Error': {'Code': 'NoSuchKey', 'Message': 'Key not found'}}, 'DeleteObject')
    s3_key = "test_prefix/doc123/test.txt"

    result = service.delete_file(s3_key)
    assert result is False
    mock_boto3_client.delete_object.assert_called_once()

def test_delete_file_unexpected_error(mock_settings_s3_configured, mock_boto3_client):
    """Test delete failure due to an unexpected exception."""
    service = S3Service(mock_settings_s3_configured)
    mock_boto3_client.delete_object.side_effect = Exception("Something broke")
    s3_key = "test_prefix/doc123/test.txt"

    result = service.delete_file(s3_key)
    assert result is False
    mock_boto3_client.delete_object.assert_called_once() 