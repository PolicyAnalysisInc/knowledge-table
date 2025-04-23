"""Pydantic models for validating responses from the LLM API."""

import logging
from typing import Any, List, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseResponseModel(BaseModel):
    """Base class for response models with common validation logic."""

    # Keep confidence and reasoning non-optional
    confidence: int = Field(
        description="The confidence score of the LLM's response, from 1 (lowest) to 10 (highest)."
    )

    reasoning: str = Field(
        description="The reasoning behind your answer. Be as thorough as necessary to justify your answer based on the context."
    )

    # Add back the all_responses field
    all_responses: Optional[List[Optional["BaseResponseModel"]]] = Field(
        default=None,
        exclude=True, # Exclude from serialization by default
        description="Internal field used by MultiCompletionService to store all parallel responses."
    )

    @classmethod
    def validate_none(cls, v: Any) -> Optional[Any]:
        """Validate if the value is None or "none"."""
        if v is None or (
            isinstance(v, str)
            and v.lower() in ["none", "not found", "null", ""]
        ):
            return None
        return v

    @field_validator("confidence", mode='before')
    @classmethod
    def validate_confidence(cls, v: Any) -> Optional[int]:
        """Validate if the confidence score is an integer between 1 and 10 or None."""
        if v is None or (isinstance(v, str) and v.lower() in ["none", "null"]):
            return None
        try:
            confidence = int(v)
            if 1 <= confidence <= 10:
                return confidence
            else:
                logger.warning(f"Confidence score {confidence} out of range (1-10). Setting to None.")
                return None
        except (ValueError, TypeError):
            logger.warning(f"Invalid confidence score value: {v}. Setting to None.")
            return None


class BoolResponseModel(BaseResponseModel):
    """Pydantic model for validating boolean responses."""

    answer: Optional[bool] = Field(
        description="The boolean answer to the query"
    )

    @field_validator("answer", mode="before")
    def validate_bool(cls, v: Any) -> Optional[bool]:
        """Validate if the value is a boolean or None."""
        v = cls.validate_none(v)
        if v is None:
            return None
        if isinstance(v, list) and len(v) == 1:
            v = v[0]
        if isinstance(v, str):
            if v.lower() == "true":
                return True
            elif v.lower() == "false":
                return False
        elif isinstance(v, bool):
            return v
        raise ValueError("Must be a boolean value or None")


class IntResponseModel(BaseResponseModel):
    """Pydantic model for validating integer responses."""

    answer: Optional[int] = Field(
        description="The integer answer to the query"
    )

    @field_validator("answer", mode="before")
    def validate_int(cls, v: Any) -> Optional[int]:
        """Validate if the value is an integer or None."""
        v = cls.validate_none(v)
        if v is None:
            return None
        if isinstance(v, list) and len(v) == 1:
            v = v[0]
        # Explicitly check for float type that is not an integer
        if isinstance(v, float) and not v.is_integer():
             logger.warning(f"Received float ({v}) for integer field. Truncating.")
             # Or raise ValueError("Must be an integer value, not a float") if strict
             return int(v) # Or return None / raise error
        try:
            return int(v)
        except (ValueError, TypeError):
            raise ValueError("Must be an integer value or None")


class NumberResponseModel(BaseResponseModel):
    """Pydantic model for validating numeric (float) responses."""

    answer: Optional[float] = Field(
        description="The numeric (float) answer to the query"
    )

    @field_validator("answer", mode="before")
    def validate_number(cls, v: Any) -> Optional[float]:
        """Validate if the value is a float or None."""
        v = cls.validate_none(v)
        if v is None:
            return None
        if isinstance(v, list) and len(v) == 1:
            v = v[0]
        try:
            # Attempt to convert to float
            return float(v)
        except (ValueError, TypeError):
            raise ValueError("Must be a valid number (integer or float) or None")


class ArrayResponseModel(BaseResponseModel):
    """Base class for array response models."""

    @classmethod
    def validate_array(
        cls, v: Any, max_length: Optional[int] = None
    ) -> Optional[List[Any]]:
        """Validate if the value is a list or None."""
        v = cls.validate_none(v)
        if v is None:
            return None
        if len(v) == 1 and v[0] == "None":
            return None
        if not isinstance(v, list):
            raise ValueError("Must be a list or None")
        if max_length and len(v) > max_length:
            logger.error(f"Length of response is greater than {max_length}")
            return v[:max_length]
        return v


class IntArrayResponseModel(ArrayResponseModel):
    """Pydantic model for validating integer array responses."""

    answer: Optional[List[int]] = Field(
        description="The list of integer answers to the query"
    )

    @field_validator("answer", mode="before")
    @classmethod
    def validate_int_array(
        cls, v: Any, info: ValidationInfo
    ) -> Optional[List[int]]:
        """Validate if the value is an integer array or None."""
        # Assuming rules might specify max_length for arrays
        int_rule = info.context.get("int_rule") if info.context else None # Use context
        max_length = int_rule.length if int_rule else None
        v = cls.validate_array(v, max_length)
        if v is None:
            return None
        if isinstance(v, (int, float)):
             # Allow single number to be treated as a list
             v = [v]
        validated_list = []
        for item in v:
            try:
                val = float(item) # Check if it's a number first
                if not val.is_integer():
                    logger.warning(f"Received float ({val}) in integer array. Truncating.")
                    # Or raise ValueError / skip item
                validated_list.append(int(val))
            except (ValueError, TypeError):
                 raise ValueError(f"Item '{item}' is not a valid integer")
        return validated_list


class NumberArrayResponseModel(ArrayResponseModel):
    """Pydantic model for validating number (float) array responses."""

    answer: Optional[List[float]] = Field(
        description="The list of numeric (float) answers to the query"
    )

    @field_validator("answer", mode="before")
    @classmethod
    def validate_number_array(
        cls, v: Any, info: ValidationInfo
    ) -> Optional[List[float]]:
        """Validate if the value is a number (float) array or None."""
        # Assuming rules might specify max_length for arrays
        int_rule = info.context.get("int_rule") if info.context else None # Use context
        max_length = int_rule.length if int_rule else None
        v = cls.validate_array(v, max_length)
        if v is None:
            return None
        if isinstance(v, (int, float, str)):
             # Allow single number/string to be treated as a list
             v = [v]
        validated_list = []
        for item in v:
            try:
                validated_list.append(float(item))
            except (ValueError, TypeError):
                raise ValueError(f"Item '{item}' is not a valid number (integer or float)")
        return validated_list


class StrArrayResponseModel(ArrayResponseModel):
    """Pydantic model for validating string array responses."""

    answer: Optional[List[str]] = Field(
        description="The list of string answers to the query"
    )

    @field_validator("answer", mode="before")
    @classmethod
    def validate_str_array(
        cls, v: Any, info: ValidationInfo
    ) -> Optional[List[str]]:
        """Validate if the value is a string array or None."""
        # Use context instead of data for Pydantic v2 compatibility
        str_rule = info.context.get("str_rule") if info.context else None
        int_rule = info.context.get("int_rule") if info.context else None
        max_length = int_rule.length if int_rule else None
        v = cls.validate_array(v, max_length)
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        if not all(isinstance(item, str) for item in v):
            raise ValueError("All items must be strings")
        if str_rule and str_rule.type == "must_return" and str_rule.options:
            return [i for i in v if i in str_rule.options]
        return v


class StrResponseModel(BaseResponseModel):
    """Pydantic model for validating string responses."""

    answer: Optional[str] = Field(description="The string answer to the query")

    @field_validator("answer", mode="before")
    @classmethod
    def validate_str(cls, v: Any, info: ValidationInfo) -> Optional[str]:
        """Validate if the value is a string or None."""
        v = cls.validate_none(v)
        if v is None:
            return None
        if isinstance(v, list) and len(v) == 1:
            v = v[0]
        if not isinstance(v, str):
            raise ValueError("Must be a string")
        # Use context instead of data
        str_rule = info.context.get("str_rule") if info.context else None
        if (
            str_rule
            and str_rule.type == "must_return"
            and str_rule.options is not None
            and v not in str_rule.options
        ):
            return None
        return v


class KeywordsResponseModel(ArrayResponseModel):
    """Pydantic model for validating keyword responses."""

    keywords: Optional[List[str]] = Field(
        description="The extracted keywords from the query"
    )

    @field_validator("keywords", mode="before")
    def validate_keywords(cls, v: Any) -> Optional[List[str]]:
        """Validate if the value is a string array or None."""
        return cls.validate_array(v)


class SubQueriesResponseModel(ArrayResponseModel):
    """Pydantic model for validating sub-query responses."""

    sub_queries: Optional[List[str]] = Field(
        description="The decomposed sub-queries"
    )

    @field_validator("sub_queries", mode="before")
    def validate_sub_queries(cls, v: Any) -> Optional[List[str]]:
        """Validate if the value is a string array or None."""
        return cls.validate_array(v)


class SchemaRelationship(BaseModel):
    """Pydantic model for a schema relationship."""

    head: str = Field(description="The head entity of the relationship")
    relation: str = Field(
        description="The relation between the head and tail entities"
    )
    tail: str = Field(description="The tail entity of the relationship")


class SchemaResponseModel(ArrayResponseModel):
    """Pydantic model for validating schema responses."""

    relationships: Optional[List[SchemaRelationship]] = Field(
        description="The relationships in the schema"
    )

    @field_validator("relationships", mode="before")
    def validate_relationships(
        cls, v: Any
    ) -> Optional[List[SchemaRelationship]]:
        """Validate if the value is a schema relationship array or None."""
        return cls.validate_array(v)
