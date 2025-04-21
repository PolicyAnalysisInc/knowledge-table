"""Document model."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class Document(BaseModel):
    """Document model."""

    id: str
    name: str
    author: str
    tag: str
    page_count: int


class DocumentMetadata(BaseModel):
    """Model for storing document metadata in MongoDB."""

    id: str = Field(..., alias="_id")  # Use document_id as MongoDB _id
    filename: str
    s3_key: Optional[str] = None
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    content_type: Optional[str] = None

    model_config = {
        "populate_by_name": True,  # Allow using '_id' in constructor
        "json_encoders": {datetime: lambda dt: dt.isoformat()}
    }
