# 🧠 AI vs Real Image Classifier

> Detect whether an image is AI-generated or a real photograph using deep learning.

Built with **EfficientNetV2-S**, two-phase transfer learning, Test-Time Augmentation (TTA), and served via a **FastAPI** REST API — fully containerized with Docker and automated via GitHub Actions.

---

## 📌 Problem Statement

With the rapid rise of AI image generation tools (Midjourney, DALL·E, Stable Diffusion), it is increasingly difficult for humans to distinguish AI-generated images from real photographs. This project builds an automated binary classifier that detects whether a given image is **AI-generated** or a **real photograph**, with practical deployment via a REST API.

---

## 📁 Project Structure

```
ai-vs-real-classifier/
├── README.md
├── requirements.txt
├── Dockerfile
├── config.yaml
├── .github/
│   └── workflows/
│       └── main.yml              # CI/CD pipeline
├── data/
│   ├── README.md
│   └── dataset_link.txt          # Kaggle dataset link
├── src/
│   ├── data_loader.py            # Dataset loading & train/val/test split
│   ├── preprocess.py             # Image preprocessing & augmentation
│   ├── features.py               # Feature extraction helpers
│   ├── train.py                  # Two-phase model training
│   ├── evaluate.py               # Metrics, confusion matrix, ROC curve
│   ├── predict.py                # Single-image inference with TTA
│   └── utils.py                  # Logging, seeding, config helpers
├── pipeline/
│   └── pipeline.py               # Single entry point — end-to-end ML workflow
├── models/
│   ├── model_v1.keras            # Trained model artifact
│   └── class_names.json          # Class config, thresholds, metadata
├── app/
│   ├── app.py                    # FastAPI application + startup
│   ├── routes.py                 # API route definitions
│   └── schema.py                 # Pydantic request/response schemas
├── frontend/
│   ├── index.html                # Upload UI
│   ├── style.css                 # Styling
│   └── script.js                 # API calls + display logic
└── logs/
    └── app.log                   # Runtime logs (predictions + errors)
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.9+
- pip
- Docker (for containerized deployment)

### Install Dependencies

```bash
git clone https://github.com/anushaanaik/ai-vs-real-classifier.git
cd ai-vs-real-classifier
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

The entire ML workflow is executed through a **single entry point**:

```bash
python pipeline/pipeline.py
```

This script automatically orchestrates:
1. Data loading from the dataset directory
2. Preprocessing and augmentation
3. Feature engineering
4. Two-phase model training (head warmup → fine-tuning)
5. Saving the trained model artifact

**Output after pipeline completes:**
```
models/model_v1.keras       ← trained model
models/class_names.json     ← class config and thresholds
logs/app.log                ← training logs
```

No manual intervention is required. The pipeline runs from raw data to a deployable model in one command.

---

## 🌐 Running the API

### Start the development server

```bash
uvicorn app.app:app --reload --port 8000
```

### Access the auto-generated API docs

```
http://localhost:8000/docs
```

---

## 🐳 Docker

### Build the image

```bash
docker build -t ai-vs-real .
```

### Run the container

```bash
docker run -p 8000:8000 ai-vs-real
```

### Access API docs

```
http://localhost:8000/docs
```

The containerized API is fully self-contained and runs identically across all environments.

---

## 📡 API Usage

The API exposes two prediction endpoints.

---

### `POST /predict` — JSON body *(primary endpoint)*

Accepts a Base64-encoded image in a JSON body, validated by `schema.py`.

**Request:**
```bash
B64=$(base64 -w 0 photo.jpg)

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\": \"$B64\", \"filename\": \"photo.jpg\", \"tta_steps\": 8}"
```

**Request body schema:**
```json
{
  "image_b64": "<base64-encoded image string>",
  "filename":  "photo.jpg",
  "tta_steps": 8
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_b64` | string | ✅ | Base64-encoded image (JPEG, PNG, WebP, GIF) |
| `filename` | string | ❌ | Original filename (default: `image.jpg`) |
| `tta_steps` | int (1–32) | ❌ | Test-Time Augmentation steps (default: 8) |

---

### `POST /predict/upload` — File upload *(browser / Swagger UI)*

Accepts a multipart file upload. Useful for quick testing via the `/docs` interface.

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/predict/upload \
  -H "accept: application/json" \
  -F "file=@photo.jpg;type=image/jpeg"
```

---

### Response (both endpoints)

```json
{
  "label":      "AI",
  "confidence": 0.87,
  "ai_prob":    0.87,
  "real_prob":  0.13,
  "tta_steps":  8,
  "filename":   "photo.jpg"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `label` | `"AI"` \| `"REAL"` \| `"UNCERTAIN"` | Predicted class |
| `confidence` | float 0–1 | Confidence score of the winning class |
| `ai_prob` | float 0–1 | Probability the image is AI-generated |
| `real_prob` | float 0–1 | Probability the image is a real photograph |
| `tta_steps` | int | Number of TTA passes used |
| `filename` | string | Filename echoed from the request |

---

### Other Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root health check |
| `GET` | `/health` | Liveness probe |
| `GET` | `/config` | Returns model configuration and metadata |

---

## 📋 Logging

All prediction requests and runtime errors are logged to `logs/app.log` using Python's built-in `logging` framework.

**Sample log entries:**
```
2025-01-15 10:23:44  INFO     AI vs Real Classifier API starting up...
2025-01-15 10:23:45  INFO     Model loaded successfully and ready to serve.
2025-01-15 10:23:51  INFO     Prediction | file=photo.jpg | label=AI | confidence=0.8712
2025-01-15 10:24:03  ERROR    Prediction error: Model not found at models/model_v1.keras
```

---

## 🏗️ Model Summary

### Architecture — EfficientNetV2-S

| Component | Detail |
|-----------|--------|
| Backbone | EfficientNetV2-S pretrained on ImageNet |
| Input size | 224 × 224 × 3 |
| Preprocessing | Built-in (`include_preprocessing=True`) — input range [0, 255] |
| Head | GlobalAvgPool → BN → Dense(512) → BN → Dropout(0.45) → Dense(256) → BN → Dropout(0.35) → Sigmoid |
| Output | Single neuron — probability of REAL class |
| Precision | Mixed FP16 training, float32 output layer |

### Training Strategy

| Phase | Description | LR | Epochs |
|-------|-------------|-----|--------|
| Phase 1 | Head-only training, base frozen | 1e-3 (cosine decay) | 10 |
| Phase 2 | Top 40 base layers unfrozen, BN layers frozen | 2e-5 (warmup + cosine) | 12 |

### Inference

- **TTA:** 8 passes with random horizontal flip per pass — final prediction is the mean probability across all passes
- **Confidence threshold:** 0.60 — predictions below threshold on both classes return `"UNCERTAIN"`

### Results

| Metric | Validation | Test |
|--------|-----------|------|
| Accuracy | ~71% | ~71% |
| AUC | 0.7831 | 0.7825 |

**Confusion Matrix (Validation set):**

| | Predicted AI | Predicted REAL |
|--|---|---|
| **True AI** | 737 ✅ | 163 ❌ |
| **True REAL** | 348 ❌ | 550 ✅ |

---

## 🔧 Configuration

Edit `config.yaml` to adjust runtime parameters:

```yaml
img_size: 224
batch_size: 128
confidence_threshold: 0.60
tta_steps: 8
dataset_root: data/dataset
model_path: models/model_v1.keras
```

---

## 🔄 CI/CD

A GitHub Actions pipeline is defined in `.github/workflows/main.yml` and triggers on every push to `main`. It builds the Docker image to verify the container compiles and runs correctly in a clean environment.
