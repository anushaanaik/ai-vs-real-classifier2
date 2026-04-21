import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.routes import router
from src.utils import get_logger

logger = get_logger(log_file="logs/app.log")

app = FastAPI(
    title="AI vs Real Image Classifier",
    description=(
        "Detect whether an image is AI-generated or a real photograph "
        "using EfficientNetV2-S with Test-Time Augmentation (TTA)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)

# ✅ THIS is the only change — root now serves index.html
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/config")
async def config():
    try:
        from src.predict import _load_config
        cfg = _load_config()
        return cfg
    except Exception as e:
        return {"error": str(e)}

@app.on_event("startup")
async def startup_event():
    logger.info("AI vs Real Classifier API starting up...")
    try:
        from src.predict import _load_model, _load_config
        _load_config()
        _load_model()
        logger.info("Model loaded successfully and ready to serve.")
    except Exception as e:
        logger.warning(
            "Model not yet available (run `python pipeline/pipeline.py` first): %s", e
        )