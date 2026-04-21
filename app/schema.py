"""
schema.py — Pydantic schemas for AI vs Real Image Classifier API.
Used by POST /predict to validate JSON input and structure output.
"""

import base64
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# REQUEST
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    JSON body for POST /predict.

    Example request:
        {
            "image_b64": "<base64-encoded image bytes>",
            "filename":  "photo.jpg",
            "tta_steps": 8
        }
    """

    image_b64: str = Field(
        ...,
        description=(
            "Base64-encoded image. Supported formats: JPEG, PNG, WebP. "
            "You may include a data-URI prefix (e.g. 'data:image/jpeg;base64,...') "
            "or send the raw Base64 string."
        ),
    )
    filename: str = Field(
        default="image.jpg",
        description="Original filename — used in logs and the response.",
    )
    tta_steps: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Test-Time Augmentation steps (1–32). Higher = more accurate, slower.",
    )

    @field_validator("image_b64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        # Strip data-URI prefix if present
        if "," in v:
            v = v.split(",", 1)[1]
        try:
            decoded = base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("image_b64 is not valid Base64.")
        if len(decoded) < 100:
            raise ValueError("Decoded image is too small — may be corrupt or empty.")
        return v

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        ext = ("." + v.rsplit(".", 1)[-1].lower()) if "." in v else ""
        if ext not in allowed_exts:
            raise ValueError(
                f"Unsupported extension '{ext}'. Allowed: {', '.join(sorted(allowed_exts))}"
            )
        return v


# ─────────────────────────────────────────────
# RESPONSE
# ─────────────────────────────────────────────

class PredictResponse(BaseModel):
    """
    JSON returned by POST /predict and POST /predict/upload.

    Example response:
        {
            "label":      "AI",
            "confidence": 0.87,
            "ai_prob":    0.87,
            "real_prob":  0.13,
            "tta_steps":  8,
            "filename":   "photo.jpg"
        }
    """

    label: Literal["AI", "REAL", "UNCERTAIN"] = Field(
        ...,
        description="Predicted class: 'AI' (generated), 'REAL' (photograph), or 'UNCERTAIN'.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence of the winning class (0.0 – 1.0).",
    )
    ai_prob: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability the image is AI-generated.",
    )
    real_prob: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability the image is a real photograph.",
    )
    tta_steps: int = Field(
        ...,
        description="Number of TTA steps used during inference.",
    )
    filename: str = Field(
        ...,
        description="Filename echoed from the request.",
    )