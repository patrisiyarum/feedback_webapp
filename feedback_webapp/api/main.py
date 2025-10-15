from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import tensorflow_text  # Registers the necessary ops for the model.
import tensorflow_hub as hub
from keras.layers import TFSMLayer  # ✅ Import added for SavedModel compatibility
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import os

# --- 1. Initialize the FastAPI App ---
app = FastAPI(
    title="Feedback Categorization API",
    description="An API to classify user feedback into main and sub-categories.",
    version="1.0.0"
)

# --- 2. Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Change "*" to your frontend URL in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Load Model and Class Files on Startup ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'two_layer_categorization_model_fixed')
MAIN_CLASSES_PATH = os.path.join(os.path.dirname(__file__), '..', 'main_category_classes.json')
SUB_CLASSES_PATH = os.path.join(os.path.dirname(__file__), '..', 'subcategory_classes.json')

model = None
main_category_classes = []
subcategory_classes = []

@app.on_event("startup")
def load_model_and_classes():
    """Load the TensorFlow model and JSON class files once when the API starts."""
    global model, main_category_classes, subcategory_classes
    try:
        if os.path.exists(MODEL_PATH):
            # ✅ Use TFSMLayer instead of load_model() for SavedModel directories
            model = TFSMLayer(
                MODEL_PATH,
                call_endpoint="serving_default"  # Change if your endpoint differs
            )
            print("✅ TensorFlow model loaded successfully via TFSMLayer.")
        else:
            raise FileNotFoundError(f"Model directory not found at {MODEL_PATH}")

        with open(MAIN_CLASSES_PATH, 'r') as f:
            main_category_classes = json.load(f)
            print(f"✅ Loaded {len(main_category_classes)} main categories.")

        with open(SUB_CLASSES_PATH, 'r') as f:
            subcategory_classes = json.load(f)
            print(f"✅ Loaded {len(subcategory_classes)} sub-categories.")

    except Exception as e:
        print("❌ Critical Error: Could not load model or class files. API will not be functional.")
        print(f"Error details: {e}")

# --- 4. Define API Schemas ---
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    main_category: str
    sub_category: str
    main_category_confidence: float
    sub_category_confidence: float

# --- 5. Endpoints ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Feedback Categorization API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Return predicted main and sub-categories with confidence scores."""
    if model is None or not main_category_classes or not subcategory_classes:
        raise HTTPException(status_code=503, detail="Model unavailable. Check server logs.")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        # The TFSMLayer behaves like a callable model
        predictions = model([request.text])

        # predictions is expected to be a list/tuple of [main_category_preds, sub_category_preds]
        main_category_preds = predictions[0][0]
        sub_category_preds = predictions[1][0]

        main_pred_index = np.argmax(main_category_preds)
        sub_pred_index = np.argmax(sub_category_preds)

        main_confidence = float(main_category_preds[main_pred_index])
        sub_confidence = float(sub_category_preds[sub_pred_index])

        main_category_result = main_category_classes[main_pred_index]
        sub_category_result = subcategory_classes[sub_pred_index]

        return PredictionResponse(
            main_category=main_category_result,
            sub_category=sub_category_result,
            main_category_confidence=main_confidence,
            sub_category_confidence=sub_confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
