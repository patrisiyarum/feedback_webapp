from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import tensorflow_text  # required for ops
import tensorflow_hub as hub
from keras.layers import TFSMLayer
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import os

# ============================================================
# 1️⃣ Initialize FastAPI
# ============================================================
app = FastAPI(
    title="Feedback Categorization API",
    description="An API to classify user feedback into main and sub-categories.",
    version="2.1.0"
)

# ============================================================
# 2️⃣ Enable CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 3️⃣ File Paths
# ============================================================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "two_layer_categorization_model_fixed")
MAIN_CLASSES_PATH = os.path.join(BASE_DIR, "..", "main_category_classes.json")
SUB_CLASSES_PATH = os.path.join(BASE_DIR, "..", "subcategory_classes.json")

model = None
input_key = "text"
main_category_classes = []
subcategory_classes = []

# ============================================================
# 4️⃣ Load Model and Labels
# ============================================================
@app.on_event("startup")
def load_model_and_classes():
    global model, input_key, main_category_classes, subcategory_classes
    try:
        # Load model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model path not found at {MODEL_PATH}")

        model = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
        print("✅ Model loaded successfully from:", MODEL_PATH)

        # Try to detect input key safely
        try:
            sigs = getattr(model, "signatures", None)
            if sigs:
                sig = sigs.get("serving_default")
                if sig:
                    key = list(sig.structured_input_signature[1].keys())[0]
                    input_key = key
        except Exception:
            # fallback if no signatures (new Keras)
            input_key = "text"

        print(f"✅ Using input key: '{input_key}'")

        # Load class label files
        with open(MAIN_CLASSES_PATH) as f:
            main_category_classes[:] = json.load(f)
        with open(SUB_CLASSES_PATH) as f:
            subcategory_classes[:] = json.load(f)

        print(f"✅ Loaded {len(main_category_classes)} main + {len(subcategory_classes)} subcategories")

    except Exception as e:
        print("❌ Error loading model or classes:", e)

# ============================================================
# 5️⃣ Schemas
# ============================================================
class PredictionRequest(BaseModel):
    text: str

# ============================================================
# 6️⃣ Health Check
# ============================================================
@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model else "unavailable",
        "model_loaded": model is not None,
        "input_key": input_key,
    }

# ============================================================
# 7️⃣ Predict Endpoint
# ============================================================
@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        text_tensor = tf.constant([request.text])
        print(f"🧠 Predicting with key '{input_key}'")

        # Support both input styles
        try:
            predictions = model(**{input_key: text_tensor})
        except TypeError:
            # fallback: some models accept positional arg
            predictions = model(text_tensor)

        # Support dict or tuple outputs
        if isinstance(predictions, dict):
            outputs = list(predictions.values())
        else:
            outputs = predictions

        main_preds = outputs[0][0].numpy()
        sub_preds = outputs[1][0].numpy()

        main_results = [
            {"label": main_category_classes[i], "probability": float(main_preds[i]) * 100}
            for i in range(len(main_preds))
        ]
        sub_results = [
            {"label": subcategory_classes[i], "probability": float(sub_preds[i]) * 100}
            for i in range(len(sub_preds))
        ]

        main_results.sort(key=lambda x: x["probability"], reverse=True)
        sub_results.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "mainPredictions": main_results,
            "subPredictions": sub_results
        }

    except Exception as e:
        print("❌ Prediction failed:", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# ============================================================
# 8️⃣ Root Endpoint
# ============================================================
@app.get("/")
def root():
    return {"status": "ok", "message": "Feedback Categorization API is running."}
