import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parent

BREAST_ARTIFACT_PATH = ROOT / "breast_cancer_model.joblib"
BRAIN_MODEL_PATH = ROOT / "model.h5"

FRONTEND_DIR = ROOT / "frontend"

IMG_SIZE = (128, 128)
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def _load_breast_artifact() -> dict[str, Any]:
    if not BREAST_ARTIFACT_PATH.is_file():
        raise FileNotFoundError(f"Missing {BREAST_ARTIFACT_PATH}")
    artifact = joblib.load(BREAST_ARTIFACT_PATH)
    if not isinstance(artifact, dict) or "model" not in artifact or "feature_columns" not in artifact:
        raise ValueError(
            "breast_cancer_model.joblib must be a dict with keys: 'model' and 'feature_columns'"
        )
    return artifact


def _load_brain_model():
    if not BRAIN_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing {BRAIN_MODEL_PATH}")
    return tf.keras.models.load_model(str(BRAIN_MODEL_PATH))


app = FastAPI(title="Medical Anomaly Detection API")

try:
    _breast_artifact = _load_breast_artifact()
    _breast_model = _breast_artifact["model"]
    _breast_feature_columns = list(_breast_artifact["feature_columns"])
except Exception as e:  # noqa: BLE001 (keep startup resilient and error visible)
    _breast_artifact = None
    _breast_model = None
    _breast_feature_columns = None
    _breast_startup_error = str(e)
else:
    _breast_startup_error = None

try:
    _brain_model = _load_brain_model()
except Exception as e:  # noqa: BLE001
    _brain_model = None
    _brain_startup_error = str(e)
else:
    _brain_startup_error = None


def _parse_csv_row(csv_row: str) -> list[float]:
    # Expected: id,diagnosis + 30 numeric features
    parts = [p.strip() for p in (csv_row or "").strip().split(",") if p.strip() != ""]
    if len(parts) != 32:
        raise ValueError(f"Expected 32 comma-separated values, got {len(parts)}")
    features_raw = parts[2:]
    features = []
    for i, raw in enumerate(features_raw):
        try:
            features.append(float(raw))
        except ValueError as e:
            raise ValueError(f"Feature #{i + 1} is not a number: {raw}") from e
    return features


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "breast_model_loaded": _breast_model is not None,
        "brain_model_loaded": _brain_model is not None,
        "breast_error": _breast_startup_error,
        "brain_error": _brain_startup_error,
    }


@app.post("/api/breast-cancer")
async def breast_cancer(payload: dict[str, Any]):
    if _breast_model is None or _breast_feature_columns is None:
        raise HTTPException(status_code=500, detail=f"Breast cancer model not loaded: {_breast_startup_error}")

    features = payload.get("features")
    csv_row = payload.get("csv_row")

    if features is None and csv_row is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'features' (30 floats) or 'csv_row' (id,diagnosis + 30 features).",
        )

    if features is None:
        try:
            features = _parse_csv_row(str(csv_row))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(e)) from e

    if not isinstance(features, list) or len(features) != 30:
        raise HTTPException(
            status_code=400,
            detail=f"'features' must be a list of 30 numbers (got {type(features)} len={len(features) if isinstance(features, list) else 'n/a'})",
        )

    try:
        x = [float(v) for v in features]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse 'features' as floats: {e}") from e

    df = pd.DataFrame([x], columns=_breast_feature_columns)

    try:
        proba = float(_breast_model.predict_proba(df)[:, 1][0])
        pred = int(_breast_model.predict(df)[0])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e

    return JSONResponse(
        {
            "prob_malignant": proba,
            "predicted_malignant": pred,
            "label": "malignant" if pred == 1 else "benign",
        }
    )


@app.post("/api/brain-tumor")
async def brain_tumor(image: UploadFile = File(...)):
    if _brain_model is None:
        raise HTTPException(status_code=500, detail=f"Brain tumor model not loaded: {_brain_startup_error}")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image.")

    raw = await image.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}") from e

    img = img.resize(IMG_SIZE)
    x = (np.array(img) / 255.0).reshape(1, 128, 128, 3)

    try:
        pred = _brain_model.predict(x)
        idx = int(np.argmax(pred))
        label = CLASS_NAMES[idx]
        confidence = float(pred[0][idx]) * 100.0
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e

    return JSONResponse({"label": label, "confidence": confidence})


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/")
def index_fallback():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(str(index_path))
    raise HTTPException(status_code=404, detail="frontend/index.html not found")

