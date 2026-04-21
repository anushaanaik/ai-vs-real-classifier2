"""
routes.py — FastAPI route definitions for the AI vs Real classifier API.

Routes:
  POST /predict        → JSON body (schema-validated) → prediction JSON
  POST /predict/upload → multipart file upload → prediction JSON
  GET  /config         → Return model config/metadata
  GET  /health         → Liveness check
"""
import base64
import io
import os
import sys

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.predict import predict_image
from src.utils import get_logger, load_class_config
from app.schema import PredictRequest, PredictResponse

logger = get_logger(log_file="logs/app.log")

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_UPLOAD_MB = 10


# ─── Shared inference helper ──────────────────────────────────────────────────

def _run_inference(pil_img: Image.Image, filename: str, tta_steps: int = 8) -> dict:
    """
    Call predict_image and attach the filename to the result dict.

    Parameters
    ----------
    pil_img   : PIL Image (RGB)
    filename  : original filename for logging / response echo
    tta_steps : number of TTA passes (passed through from request)
    """
    try:
        result = predict_image(
            pil_img,
            model_path="models/model_v1.keras",
            config_path="models/class_names.json",
            tta_steps=tta_steps,
            verbose=False,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run `python pipeline/pipeline.py` first.",
        )
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    result["filename"] = filename
    logger.info(
        "Prediction | file=%s | label=%s | confidence=%.4f | tta=%d",
        filename, result["label"], result["confidence"], tta_steps,
    )
    return result


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok"}


# ─── Model config ─────────────────────────────────────────────────────────────

@router.get("/config")
async def get_config():
    try:
        cfg = load_class_config("models/class_names.json")
        return JSONResponse(content=cfg)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model config not found. Run training first.",
        )


# ─── Prediction — JSON body (schema-validated) ────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest):
    """
    Upload an image **as Base64 JSON** and receive a prediction.

    Request body (application/json):
    ```json
    {
      "image_b64": "<base64-encoded image>",
      "filename":  "photo.jpg",
      "tta_steps": 8
    }
    ```

    Returns:
    ```json
    {
      "label":      "AI" | "REAL" | "UNCERTAIN",
      "confidence": 0.87,
      "ai_prob":    0.87,
      "real_prob":  0.13,
      "tta_steps":  8,
      "filename":   "photo.jpg"
    }
    ```
    """
    # Decode base64 → PIL image
    raw_b64 = body.image_b64
    if "," in raw_b64:                          # strip data-URI prefix
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_b64)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")

    result = _run_inference(pil_img, body.filename, tta_steps=body.tta_steps)
    return PredictResponse(**result)


# ─── Prediction — multipart file upload (kept for browser/curl convenience) ───

@router.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(file: UploadFile = File(...)):
    """
    Upload an image as **multipart/form-data** (browser / curl convenience).

    Uses the default TTA steps from `class_names.json` (usually 8).
    For custom TTA, use POST /predict with a JSON body instead.

    Returns identical output to POST /predict.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 ** 2)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max {MAX_UPLOAD_MB} MB.",
        )

    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {e}")

    # /predict/upload uses config default TTA (None → predict_image reads from class_names.json)
    result = _run_inference(pil_img, file.filename, tta_steps=8)
    return PredictResponse(**result)