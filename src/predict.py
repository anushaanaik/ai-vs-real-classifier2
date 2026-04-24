from __future__ import annotations

import argparse
import json
import os
import numpy as np
import tempfile
import uuid
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- CLOUD API IMPORTS ---
from fastapi import FastAPI, File, UploadFile, HTTPException
import boto3

# ─── Lazy model loading ─────────────────────────────────────────

_model = None
_config: dict | None = None

DEFAULT_MODEL_PATH = "models/model_v1.keras"
DEFAULT_CONFIG_PATH = "models/class_names.json"

def _load_model(model_path: str = DEFAULT_MODEL_PATH):
    global _model
    if _model is not None:
        return _model

    import keras
    import tensorflow as tf

    # ── Strategy 1: Keras 3 direct load ─────────────────────────
    try:
        keras.mixed_precision.set_global_policy("mixed_float16")
        _model = keras.models.load_model(model_path, compile=False)
        return _model
    except Exception as e:
        print("⚠️ Direct load failed, trying fallback...", str(e))

    # ── Strategy 2: Rebuild model + load weights ─────────────────
    try:
        import zipfile
        import tempfile

        tf.keras.mixed_precision.set_global_policy("float32")

        base = tf.keras.applications.EfficientNetV2S(
            include_top=False,
            weights=None,
            input_shape=(224, 224, 3),
            include_preprocessing=True,
        )

        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        rebuilt = tf.keras.Model(base.input, out, name="EffNetV2S_AIvsReal")

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(model_path, "r") as z:
                z.extractall(tmpdir)

            weight_files = [
                os.path.join(tmpdir, f)
                for f in os.listdir(tmpdir)
                if f.endswith(".weights.h5") or f == "model.weights.h5"
            ]

            if not weight_files:
                raise RuntimeError("No weights file found inside .keras")

            rebuilt.load_weights(weight_files[0])
            _model = rebuilt
            return _model

    except Exception as e:
        raise RuntimeError(f"❌ Model loading failed completely: {str(e)}")

def _load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    global _config
    if _config is None:
        with open(config_path, "r") as f:
            _config = json.load(f)
    return _config

# ─── Core inference ─────────────────────────────────────────

def predict_image(
    image_source,
    model_path: str = DEFAULT_MODEL_PATH,
    config_path: str = DEFAULT_CONFIG_PATH,
    tta_steps: int | None = None,
    threshold: float | None = None,
    verbose: bool = True,
) -> dict:

    from src.preprocess import preprocess_image

    cfg = _load_config(config_path)
    model = _load_model(model_path)

    img_size = cfg.get("input_size", 224)
    tta = tta_steps if tta_steps is not None else cfg.get("tta_steps_default", 8)
    thresh = threshold if threshold is not None else cfg.get("confidence_threshold", 0.60)

    probs = []

    for i in range(tta):
        arr = preprocess_image(image_source, img_size=img_size, apply_aug=(i > 0))
        arr = np.expand_dims(arr, axis=0)
        prob = float(model.predict(arr, verbose=0)[0][0])
        probs.append(prob)

    real_prob = float(np.mean(probs))
    ai_prob = 1.0 - real_prob

    if real_prob >= thresh:
        label, confidence = "REAL", real_prob
    elif ai_prob >= thresh:
        label, confidence = "AI", ai_prob
    else:
        label, confidence = "UNCERTAIN", max(real_prob, ai_prob)

    result = {
        "label": label,
        "confidence": round(confidence, 4),
        "ai_prob": round(ai_prob, 4),
        "real_prob": round(real_prob, 4),
        "tta_steps": tta,
    }

    if verbose:
        emoji = {"AI": "🤖", "REAL": "📷", "UNCERTAIN": "❓"}[label]
        print(f"\n{emoji} {label} (confidence: {confidence:.1%})")
        
    return result

# ─── FASTAPI & AWS CLOUD INTEGRATION ────────────────────────

# Initialize FastAPI app (this is what uvicorn will run)
# The root_path tells Swagger UI to prepend /prod to all internal API calls
app = FastAPI(
    title="AI vs Real Image Classifier - Cloud Edition",
    root_path="/prod"
)
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running!"}
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("frontend/index.html")


# Initialize AWS clients
s3_client = boto3.client('s3', region_name='us-east-1')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

BUCKET_NAME = 'ai-classifier-images-2026'
DYNAMO_TABLE_NAME = 'Predictions'

@app.post("/predict")
async def api_predict(file: UploadFile = File(...)):
    """
    Cloud API Endpoint:
    1. Receives image from user
    2. Uploads directly to Amazon S3
    3. Runs ML Model prediction
    4. Logs metadata to Amazon DynamoDB
    5. Returns JSON response
    """
    try:
        # 1. Generate unique identifiers
        image_id = str(uuid.uuid4())
        file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        s3_key = f"{image_id}.{file_ext}"

        # 2. Read file
        contents = await file.read()

        # 3. Upload to Amazon S3
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=contents,
            ContentType=file.content_type
        )
        s3_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{s3_key}"

        # 4. Save to temporary file for the ML model to read
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # 5. Run existing Prediction Logic
        ml_result = predict_image(image_source=tmp_path, verbose=False)

        # Clean up the temporary file from the container
        os.remove(tmp_path)

        # 6. Save results to Amazon DynamoDB
        table = dynamodb.Table(DYNAMO_TABLE_NAME)
        table.put_item(
            Item={
                'image_id': image_id,
                's3_url': s3_url,
                'label': ml_result['label'],
                'confidence': str(ml_result['confidence']),
                'ai_prob': str(ml_result['ai_prob']),
                'real_prob': str(ml_result['real_prob'])
            }
        )

        # 7. Return complete payload to user
        ml_result["image_id"] = image_id
        ml_result["s3_url"] = s3_url
        return ml_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── CLI (Kept for backward compatibility) ──────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--tta", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--json", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    result = predict_image(
        args.image,
        model_path=args.model,
        config_path=args.config,
        tta_steps=args.tta,
        threshold=args.threshold,
        verbose=not args.json,
    )

    if args.json:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()