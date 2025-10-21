
# FCR Feedback Categorization - Backend API

This repository contains the backend API for the "FCR Feedback Categorization Tool." It is a Python web server built with **FastAPI** that serves a **TensorFlow/Keras** machine learning model.

This backend is designed to be deployed as a standalone service (e.g., on Render) and be called by a separate frontend application (such as a React or Streamlit app).

## 🚀 Project Purpose

The purpose of this project is to automate the analysis of unstructured text feedback. It was specifically developed to address the challenge of manually processing thousands of comments from airline crew members regarding on-board meal quality and service.

The model ingests a raw text comment and classifies it into two distinct categories:
1.  **Main Category** (e.g., "Food Quality", "Catering Error")
2.  **Sub-Category** (e.g., "Taste Issues", "Incorrect Meal")

This converts unstructured, qualitative feedback into structured, actionable data at scale.

## 🧠 The Machine Learning Model

The core of this API is a fine-tuned **BERT** model.

* **Architecture:** It is a dual-output model built in TensorFlow/Keras. It uses a pre-trained BERT encoder (`bert-en-uncased-l-6-h-128-a-2`) with two separate classification "heads" (Dense layers) on top—one for predicting the main category and one for the sub-category.
* **Training:** The model was fine-tuned on a historical dataset of manually-labeled crew comments, allowing it to learn the specific language and patterns of this feedback.
* **Model Files:** The trained and exported model is stored in the `two_layer_categorization_model_fixed/` directory.

## 🛠️ Technology Stack

* **Backend:** **FastAPI**
* **ML Model:** **TensorFlow 2 / Keras**
* **Base Architecture:** **BERT** (from TensorFlow Hub)
* **Server:** **Uvicorn**

## 📁 Repository Structure

```

/
|-- api/
|   |-- main.py              \# The main FastAPI application logic
|   ` -- requirements.txt       # Python libraries for the API | |-- two_layer_categorization_model_fixed/  # The exported TensorFlow/BERT model |   |-- saved_model.pb |   |-- variables/ |    `-- assets/
|
|-- main\_category\_classes.json   \# List of all possible main categories
|-- subcategory\_classes.json     \# List of all possible subcategories
|-- runtime.txt                  \# Specifies the Python version for Render
\`-- ...

````

* `api/main.py`: This script starts the FastAPI server. On startup, it loads the TensorFlow model and category class files into memory.
* `two_layer_categorization_model_fixed/`: This is the saved TensorFlow model. It includes the BERT preprocessor and the fine-tuned encoder.
* `*.json` files: These files map the model's numerical outputs back to human-readable category names (e.g., `5` -> `"Food Quality"`).

## 🔌 API Endpoints

The API exposes several endpoints for use by a frontend application.

### `GET /health`
Checks the health of the API.
* **Purpose:** Used by the frontend to see if the server is online and the model has loaded successfully.
* **Success Response (200):**
    ```json
    {
      "status": "healthy",
      "model_loaded": true,
      "main_classes_count": 10,
      "sub_classes_count": 30
    }
    ```

### `POST /predict`
Analyzes a single piece of text.
* **Request Body:**
    ```json
    {
      "text": "The flight attendant was very helpful."
    }
    ```
* **Success Response (200):** Returns lists of all possible predictions, sorted by confidence.
    ```json
    {
      "mainPredictions": [
        { "label": "Service", "probability": 98.5 },
        { "label": "Food Quality", "probability": 1.5 }
      ],
      "subPredictions": [
        { "label": "Positive Interaction", "probability": 97.0 },
        { "label": "Attitude", "probability": 3.0 }
      ]
    }
    ```

### `POST /predict/bulk`
Analyzes a list (batch) of text comments.
* **Request Body:**
    ```json
    {
      "texts": [
        "The burger was cold.",
        "My meal was missing."
      ]
    }
    ```
* **Success Response (200):** Returns a list of prediction objects.
    ```json
    {
      "predictions": [
        {
          "mainPredictions": [ ... ],
          "subPredictions": [ ... ]
        },
        {
          "mainPredictions": [ ... ],
          "subPredictions": [ ... ]
        }
      ]
    }
    ```

### `GET /categories`
Retrieves the full lists of possible main and sub-categories.
* **Success Response (200):**
    ```json
    {
      "mainCategories": ["Service", "Food Quality", "Catering Error", ...],
      "subCategories": ["Positive Interaction", "Taste Issues", "Incorrect Meal", ...]
    }
    ```

## 🚀 Deployment (Render)

This API is configured for deployment on Render.

* **Build Command:** `pip install -r api/requirements.txt`
* **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
* **Python Version:** Specified in `runtime.txt`.
````
