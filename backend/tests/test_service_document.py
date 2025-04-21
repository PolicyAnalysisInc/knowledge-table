from unittest.mock import AsyncMock, patch, MagicMock, ANY

import pytest
from app.models.document import DocumentMetadata


@pytest.fixture
def mock_s3_service():
    service = MagicMock()
    service._is_configured.return_value = True
    service.upload_file.return_value = True
    service.delete_file.return_value = True
    service.get_object_key.return_value = "s3_test_key/doc_id/test.pdf"
    return service

@pytest.fixture
def mock_metadata_service():
    service = MagicMock()
    service._is_configured.return_value = True
    service.add_document_metadata.return_value = True
    service.delete_document_metadata.return_value = True
    service.get_s3_key_for_document.return_value = "s3_test_key/doc_id/test.pdf"
    return service


@pytest.mark.asyncio
async def test_upload_document(
    document_service, mocker, mock_s3_service, mock_metadata_service
):
    # Inject mocks
    document_service.s3_service = mock_s3_service
    document_service.metadata_service = mock_metadata_service

    # Mocks for existing dependencies
    mocker.patch.object(
        document_service, "_generate_document_id", return_value="test_id"
    )
    mocker.patch.object(document_service, "_process_document", return_value=[])
    document_service.vector_db_service.prepare_chunks = AsyncMock(
        return_value=[]
    )
    document_service.vector_db_service.upsert_vectors = AsyncMock()

    # Patch tempfile creation
    mocker.patch("tempfile.NamedTemporaryFile")
    mocker.patch("os.remove")

    # Run upload
    result = await document_service.upload_document(
        filename="test.pdf",
        file_content=b"test content",
        content_type="application/pdf",
    )

    # Assertions
    assert result == "test_id"
    document_service._process_document.assert_called_once()
    document_service.vector_db_service.prepare_chunks.assert_called_once()
    document_service.vector_db_service.upsert_vectors.assert_called_once()

    # Assert S3 and Metadata calls
    mock_s3_service.get_object_key.assert_called_once_with("test_id", "test.pdf")
    mock_s3_service.upload_file.assert_called_once_with(
        file_content=b"test content",
        s3_key="s3_test_key/doc_id/test.pdf",
        content_type="application/pdf",
    )
    mock_metadata_service.add_document_metadata.assert_called_once()
    # Check the DocumentMetadata object passed
    call_args, _ = mock_metadata_service.add_document_metadata.call_args
    metadata_arg = call_args[0]
    assert isinstance(metadata_arg, DocumentMetadata)
    assert metadata_arg.id == "test_id"
    assert metadata_arg.filename == "test.pdf"
    assert metadata_arg.s3_key == "s3_test_key/doc_id/test.pdf"
    assert metadata_arg.content_type == "application/pdf"

@pytest.mark.asyncio
async def test_upload_document_s3_failure(document_service, mocker, mock_s3_service, mock_metadata_service):
    """Test upload when S3 upload fails but metadata save proceeds (without s3 key)."""
    document_service.s3_service = mock_s3_service
    document_service.metadata_service = mock_metadata_service
    mock_s3_service.upload_file.return_value = False # Simulate S3 upload failure

    mocker.patch.object(document_service, "_generate_document_id", return_value="test_id_s3_fail")
    mocker.patch.object(document_service, "_process_document", return_value=[MagicMock()])
    document_service.vector_db_service.prepare_chunks = AsyncMock(return_value=[])
    document_service.vector_db_service.upsert_vectors = AsyncMock()
    mocker.patch("tempfile.NamedTemporaryFile")
    mocker.patch("os.remove")

    result = await document_service.upload_document("fail.txt", b"content", "text/plain")

    assert result == "test_id_s3_fail"
    mock_s3_service.upload_file.assert_called_once()
    mock_metadata_service.add_document_metadata.assert_called_once()
    # Verify s3_key is None when passed to metadata service
    call_args, _ = mock_metadata_service.add_document_metadata.call_args
    metadata_arg = call_args[0]
    assert metadata_arg.s3_key is None

@pytest.mark.asyncio
async def test_upload_document_metadata_failure(document_service, mocker, mock_s3_service, mock_metadata_service):
    """Test upload when Metadata save fails (logs error, still returns doc_id)."""
    document_service.s3_service = mock_s3_service
    document_service.metadata_service = mock_metadata_service
    mock_metadata_service.add_document_metadata.return_value = False # Simulate metadata failure

    mocker.patch.object(document_service, "_generate_document_id", return_value="test_id_meta_fail")
    mocker.patch.object(document_service, "_process_document", return_value=[MagicMock()])
    document_service.vector_db_service.prepare_chunks = AsyncMock(return_value=[])
    document_service.vector_db_service.upsert_vectors = AsyncMock()
    mocker.patch("tempfile.NamedTemporaryFile")
    mocker.patch("os.remove")

    result = await document_service.upload_document("meta_fail.txt", b"content")

    assert result == "test_id_meta_fail" # Still returns ID based on current logic
    mock_s3_service.upload_file.assert_called_once()
    mock_metadata_service.add_document_metadata.assert_called_once()

@pytest.mark.asyncio
async def test_upload_document_vector_db_failure_rollback(document_service, mocker, mock_s3_service, mock_metadata_service):
    """Test rollback is attempted if vector DB upsert fails after S3/metadata."""
    document_service.s3_service = mock_s3_service
    document_service.metadata_service = mock_metadata_service

    mocker.patch.object(document_service, "_generate_document_id", return_value="test_id_rollback")
    mocker.patch.object(document_service, "_process_document", return_value=[MagicMock()])
    document_service.vector_db_service.prepare_chunks = AsyncMock(return_value=[])
    # Simulate vector DB failure
    document_service.vector_db_service.upsert_vectors = AsyncMock(side_effect=Exception("Vector DB Error"))
    # Mock delete calls for rollback verification
    document_service.vector_db_service.delete_document = AsyncMock(return_value={"status": "success"}) # Mock rollback delete
    mock_s3_service.delete_file.return_value = True
    mock_metadata_service.delete_document_metadata.return_value = True

    mocker.patch("tempfile.NamedTemporaryFile")
    mocker.patch("os.remove")

    result = await document_service.upload_document("rollback.txt", b"content")

    assert result is None # Upload should fail
    # Verify initial actions were called
    mock_s3_service.upload_file.assert_called_once()
    mock_metadata_service.add_document_metadata.assert_called_once()
    # Verify rollback actions were called
    document_service.vector_db_service.delete_document.assert_called_once_with("test_id_rollback")
    mock_s3_service.delete_file.assert_called_once_with("s3_test_key/doc_id/test.pdf")
    mock_metadata_service.delete_document_metadata.assert_called_once_with("test_id_rollback")


@pytest.mark.asyncio
async def test_delete_document(document_service, mock_s3_service, mock_metadata_service):
    # Inject mocks
    document_service.s3_service = mock_s3_service
    document_service.metadata_service = mock_metadata_service

    # Mock vector DB delete success
    document_service.vector_db_service.delete_document = AsyncMock(
        return_value={"status": "success", "message": "Deleted from vector DB"}
    )

    result = await document_service.delete_document("test_id")

    # Assertions
    assert result["status"] == "success"
    assert "all stores" in result["message"]
    mock_metadata_service.get_s3_key_for_document.assert_called_once_with("test_id")
    mock_s3_service.delete_file.assert_called_once_with("s3_test_key/doc_id/test.pdf")
    mock_metadata_service.delete_document_metadata.assert_called_once_with("test_id")
    document_service.vector_db_service.delete_document.assert_called_once_with("test_id")

@pytest.mark.asyncio
async def test_delete_document_s3_failure(document_service, mock_s3_service, mock_metadata_service):
    """Test delete proceeds but returns partial_failure if S3 delete fails."""
    document_service.s3_service = mock_s3_service
    document_service.metadata_service = mock_metadata_service
    mock_s3_service.delete_file.return_value = False # Simulate S3 delete failure

    document_service.vector_db_service.delete_document = AsyncMock(return_value={"status": "success"})

    result = await document_service.delete_document("test_id_s3_fail")

    assert result["status"] == "partial_failure"
    assert "S3 delete failed" in result["message"]
    mock_s3_service.delete_file.assert_called_once()
    mock_metadata_service.delete_document_metadata.assert_called_once() # Should still be called
    document_service.vector_db_service.delete_document.assert_called_once() # Should still be called

@pytest.mark.asyncio
async def test_delete_document_metadata_failure(document_service, mock_s3_service, mock_metadata_service):
    """Test delete proceeds but returns partial_failure if metadata delete fails."""
    document_service.s3_service = mock_s3_service
    document_service.metadata_service = mock_metadata_service
    mock_metadata_service.delete_document_metadata.return_value = False # Simulate failure

    document_service.vector_db_service.delete_document = AsyncMock(return_value={"status": "success"})

    result = await document_service.delete_document("test_id_meta_fail")

    assert result["status"] == "partial_failure"
    assert "Metadata delete failed" in result["message"]
    mock_s3_service.delete_file.assert_called_once() # Should still be called
    mock_metadata_service.delete_document_metadata.assert_called_once()
    document_service.vector_db_service.delete_document.assert_called_once() # Should still be called

@pytest.mark.asyncio
async def test_delete_document_vector_db_failure(document_service, mock_s3_service, mock_metadata_service):
    """Test delete returns failure if vector DB delete fails."""
    document_service.s3_service = mock_s3_service
    document_service.metadata_service = mock_metadata_service

    # Simulate vector DB failure
    document_service.vector_db_service.delete_document = AsyncMock(return_value={"status": "failure", "message": "Vector DB down"})

    result = await document_service.delete_document("test_id_vec_fail")

    assert result["status"] == "failure"
    assert "Vector DB delete failed" in result["message"]
    mock_s3_service.delete_file.assert_called_once()
    mock_metadata_service.delete_document_metadata.assert_called_once()
    document_service.vector_db_service.delete_document.assert_called_once()

@pytest.mark.asyncio
async def test_delete_document_no_s3_key(document_service, mock_s3_service, mock_metadata_service):
    """Test delete skips S3 if no key found in metadata."""
    document_service.s3_service = mock_s3_service
    document_service.metadata_service = mock_metadata_service
    mock_metadata_service.get_s3_key_for_document.return_value = None # No key found

    document_service.vector_db_service.delete_document = AsyncMock(return_value={"status": "success"})

    result = await document_service.delete_document("test_id_no_key")

    assert result["status"] == "success"
    mock_metadata_service.get_s3_key_for_document.assert_called_once_with("test_id_no_key")
    mock_s3_service.delete_file.assert_not_called() # Should not be called
    mock_metadata_service.delete_document_metadata.assert_called_once()
    document_service.vector_db_service.delete_document.assert_called_once()


# --- Existing tests (need adjustment?) --- #

@pytest.mark.asyncio
async def test_delete_document_legacy_failure(document_service):
    # This test seems redundant now given the detailed failure tests above.
    # It tested the old boolean return, now it returns a dict.
    # Let's adapt it slightly or remove.
    document_service.s3_service = MagicMock(_is_configured=lambda: False) # Mock S3 not configured
    document_service.metadata_service = MagicMock(_is_configured=lambda: False) # Mock Meta not configured

    document_service.vector_db_service.delete_document = AsyncMock(
        return_value={"status": "failure", "message": "Vector DB down"} # Simulate failure
    )

    result = await document_service.delete_document("test_id")

    assert result["status"] == "failure"
    document_service.vector_db_service.delete_document.assert_called_once_with("test_id")


@pytest.mark.asyncio
async def test_upload_document_unstructured_not_available(
    document_service, mocker
):
    # This test focuses on loader logic, doesn't need S3/Meta mocks unless they interfere
    document_service.s3_service = MagicMock(_is_configured=lambda: False) # Ensure S3 doesn't interfere
    document_service.metadata_service = MagicMock(_is_configured=lambda: False) # Ensure Meta doesn't interfere

    mocker.patch.object(
        document_service, "_generate_document_id", return_value="test_id"
    )
    mocker.patch.object(document_service, "_process_document", return_value=[])
    document_service.vector_db_service.prepare_chunks = AsyncMock(
        return_value=[]
    )
    document_service.vector_db_service.upsert_vectors = AsyncMock()

    with patch(
        "app.services.loaders.unstructured_service.UNSTRUCTURED_AVAILABLE",
        False,
    ):
        result = await document_service.upload_document(
            "test.pdf", b"test content"
        )

    assert result == "test_id"
    document_service._process_document.assert_called_once()
    document_service.vector_db_service.prepare_chunks.assert_called_once_with(
        "test_id", []
    )
    document_service.vector_db_service.upsert_vectors.assert_called_once_with(
        []
    )  # Remove 'test_id' from here


@pytest.mark.asyncio
async def test_process_document_unstructured_not_available(
    document_service, mocker
):
    mocker.patch.object(
        document_service.loader_factory, "create_loader", return_value=None
    )

    with pytest.raises(
        ValueError,
        match="No loader available for configured loader type: test_loader",
    ):
        await document_service._process_document("test_file_path")
