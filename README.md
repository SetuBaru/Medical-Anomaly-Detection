# Medical Anomaly Detection (Frontend + Inference API)

This project provides a small web UI (HTML/CSS) and a local inference API to run:

- **Breast Cancer Prediction** (scikit-learn model in `breast_cancer_model.joblib`)
- **Brain Tumor Detection** (TensorFlow/Keras model in `model.h5`)

The UI lets you select which model to use, then either paste a CSV row (breast cancer) or upload an image (brain tumor).

## Requirements

- Python 3.11+ recommended
- `pip`

## Install

From the project root:

```bash
python -m pip install -r requirements.txt
```

## Run the app

Start the local server:

```bash
python -m uvicorn server:app --port 8001
```

Open the UI in your browser:

- `http://127.0.0.1:8001/`

## How to use

### 1) Breast Cancer Prediction

1. Select **Breast Cancer Prediction**
2. Paste **one CSV row** containing exactly:

`id, diagnosis, + 30 numeric features`

Example format:

```text
842302,M,17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
```

3. Click **Run breast cancer prediction**

You’ll see:

- **Prediction**: `benign` or `malignant`
- **Probability (malignant)**: a percentage

### 2) Brain Tumor Detection

1. Select **Brain Tumor Detection**
2. Upload an image (JPG/PNG)
3. Click **Run brain tumor detection**

You’ll see:

- **Prediction**: one of `glioma_tumor`, `meningioma_tumor`, `no_tumor`, `pituitary_tumor`
- **Confidence**: a percentage

## API endpoints

- `GET /api/health`
  - Returns whether both models loaded successfully.
- `POST /api/breast-cancer`
  - JSON body supports either:
    - `features`: list of 30 floats, OR
    - `csv_row`: the full row including `id,diagnosis,...`
- `POST /api/brain-tumor`
  - `multipart/form-data` with file field name: `image`

## Files

- `frontend/index.html`: UI
- `frontend/styles.css`: UI styling
- `frontend/app.js`: UI logic + API calls + result formatting
- `server.py`: FastAPI app serving UI + inference endpoints
- `breast_cancer_model.joblib`: saved breast cancer model artifact
- `model.h5`: saved brain tumor model

## Troubleshooting

- **Server takes a while to start**: TensorFlow can take 30–120 seconds to load `model.h5` the first time.
- **Port already in use**: run on a different port, e.g.:

```bash
python -m uvicorn server:app --port 8010
```

- **UI shows “Could not reach backend”**: make sure you opened the UI via `http://127.0.0.1:PORT/` (not by double-clicking the HTML file).

