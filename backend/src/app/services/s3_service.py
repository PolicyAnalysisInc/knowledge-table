"""Service for interacting with AWS S3."""

import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from app.core.config import Settings

logger = logging.getLogger(__name__)


class S3Service:
    """Handles S3 operations like uploading and deleting files."""

    def __init__(self, settings: Settings):
        """Initializes the S3 client."""
        self.settings = settings
        if (
            settings.aws_access_key_id
            and settings.aws_secret_access_key
            and settings.aws_region
            and settings.s3_bucket_name
        ):
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region,
            )
            self.bucket_name = settings.s3_bucket_name
            logger.info(f"S3 client initialized for bucket: {self.bucket_name}")
        else:
            self.s3_client = None
            self.bucket_name = None
            logger.warning(
                "S3 client not initialized. Missing S3 configuration."
            )

    def _is_configured(self) -> bool:
        """Check if S3 service is properly configured."""
        return self.s3_client is not None and self.bucket_name is not None

    def upload_file(
        self,
        file_content: bytes,
        s3_key: str,
        content_type: Optional[str] = None,
    ) -> bool:
        """
        Uploads file content to S3.

        Args:
            file_content: The raw bytes of the file.
            s3_key: The desired key (path) for the object in S3.
            content_type: Optional content type for the S3 object.

        Returns:
            True if upload was successful, False otherwise.
        """
        if not self._is_configured():
            logger.warning("S3 upload skipped: S3 service not configured.")
            # Depending on requirements, this might return True or raise an error.
            # For now, assume not configured means "don't block".
            return True

        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_content,
                **extra_args,
            )
            logger.info(f"Successfully uploaded {s3_key} to S3 bucket {self.bucket_name}")
            return True
        except ClientError as e:
            logger.error(
                f"Failed to upload {s3_key} to S3 bucket {self.bucket_name}: {e}"
            )
            return False
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during S3 upload for {s3_key}: {e}"
            )
            return False

    def delete_file(self, s3_key: str) -> bool:
        """
        Deletes a file from S3.

        Args:
            s3_key: The key (path) of the object to delete in S3.

        Returns:
            True if deletion was successful or if S3 is not configured,
            False otherwise.
        """
        if not self._is_configured():
            logger.warning("S3 delete skipped: S3 service not configured.")
            return True

        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(
                f"Successfully deleted {s3_key} from S3 bucket {self.bucket_name}"
            )
            return True
        except ClientError as e:
            # Handle specific errors like 'NoSuchKey' if needed, but often
            # deleting a non-existent key isn't a failure for idempotency.
            logger.error(
                f"Failed to delete {s3_key} from S3 bucket {self.bucket_name}: {e}"
            )
            return False
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during S3 delete for {s3_key}: {e}"
            )
            return False

    def get_object_key(self, document_id: str, filename: str) -> str:
        """
        Constructs the S3 object key for a given document.

        Args:
            document_id: The unique ID of the document.
            filename: The original filename of the uploaded document.

        Returns:
            The constructed S3 object key.
        """
        prefix = self.settings.s3_prefix.strip("/")
        # Basic sanitization, consider more robust slugification if needed
        safe_filename = "".join(c for c in filename if c.isalnum() or c in (".", "-", "_"))
        return f"{prefix}/{document_id}/{safe_filename}" 