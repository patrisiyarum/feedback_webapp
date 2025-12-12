# FCR Feedback Categorization - Backend API

This repository contains the backend API for the **FCR Feedback Categorization Tool**. It is a Python web server built with **FastAPI** that serves a **TensorFlow/Keras** machine learning model for automated text feedback classification.

This backend is designed to be deployed as a standalone service (e.g., on Render) and can be called by any frontend application.

## Project Purpose

The purpose of this project is to automate the analysis of unstructured text feedback. It was specifically developed to address the challenge of manually processing thousands of comments from airline crew members regarding on-board meal quality and service.

The model ingests raw text comments and classifies them into **subcategories** (e.g., "Taste Issues", "Incorrect Meal", "Positive Interaction"), converting unstructured qualitative feedback into structured, actionable data at scale.

## The Machine Learning Model

The core of this API is a fine-tuned **BERT** model for subcategory classification.

* **Architecture:** Built in TensorFlow/Keras using a pre-trained BERT encoder (`bert-en-uncased-l-6-h-128-a-2`) with additional keyword features and a Dense classification layer
* **Model File:** `api/subcategory_model_augmented.keras`
* **Class Labels:** `api/subcategory_classes.json` contains the mapping of model outputs to category names
* **Features:** The model combines BERT text embeddings with keyword-based features for improved accuracy

## Technology Stack

* **Backend Framework:** FastAPI
* **ML Framework:** TensorFlow 2 / Keras
* **Base Model:** BERT (from TensorFlow Hub)
* **Server:** Uvicorn
* **Python Version:** 3.11.9 (specified in `runtime.txt`)

## Repository Structure

```
feedback_webapp/
├── api/
│   ├── main.py                           # FastAPI application
│   ├── requirements.txt                   # Python dependencies
│   ├── subcategory_model_augmented.keras  # Trained BERT model
│   └── subcategory_classes.json           # Category label mappings
├── runtime.txt                            # Python version for deployment
└── README.md                              # This file
```

### Key Files

* **`api/main.py`**: The FastAPI server that loads the model and exposes prediction endpoints
* **`api/subcategory_model_augmented.keras`**: The trained TensorFlow model (BERT-based)
* **`api/subcategory_classes.json`**: Maps model output indices to human-readable category names
* **`api/requirements.txt`**: Lists all Python dependencies needed to run the API
* **`runtime.txt`**: Specifies the Python version for deployment platforms

## API Endpoints

### `GET /`
Root endpoint to check if the API is online.
* **Response:**
    ```json
    {
      "status": "online",
      "model": "Subcategory Classification (Local LFS)"
    }
    ```

### `GET /health`
Health check endpoint to verify the API and model are loaded correctly.
* **Response (200):**
    ```json
    {
      "status": "healthy",
      "sub_classes_count": 8
    }
    ```

### `POST /predict`
Analyzes a single piece of text feedback.
* **Request Body:**
    ```json
    {
      "text": "The chicken was cold and tasted bad."
    }
    ```
* **Response (200):**
    ```json
    {
      "subPredictions": [
        { "label": "Taste Issues", "probability": 87.3 },
        { "label": "Temperature Issues", "probability": 10.2 },
        { "label": "Quality Issues", "probability": 2.5 }
      ]
    }
    ```

### `POST /predict/bulk`
Analyzes multiple text comments in a single request.
* **Request Body:**
    ```json
    {
      "texts": [
        "The burger was cold.",
        "My meal was missing.",
        "Excellent service!"
      ]
    }
    ```
* **Response (200):**
    ```json
    {
      "predictions": [
        {
          "subPredictions": [
            { "label": "Temperature Issues", "probability": 92.1 },
            ...
          ]
        },
        {
          "subPredictions": [
            { "label": "Incorrect Meal", "probability": 88.5 },
            ...
          ]
        },
        {
          "subPredictions": [
            { "label": "Positive Interaction", "probability": 95.7 },
            ...
          ]
        }
      ]
    }
    ```

### `GET /categories`
Retrieves the full list of possible subcategories.
* **Response (200):**
    ```json
    {
      "subCategories": [
        "Taste Issues",
        "Temperature Issues",
        "Incorrect Meal",
        "Positive Interaction",
        ...
      ]
    }
    ```

## Local Development

### Prerequisites
- Python 3.11.9
- pip

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r api/requirements.txt
   ```

2. **Run the server:**
   ```bash
   cd feedback_webapp
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Test the API:**
   ```bash
   curl http://localhost:8000/health
   ```

4. **View interactive API docs:**
   - Open your browser to `http://localhost:8000/docs`

## Deployment on Render

This API is configured for deployment on [Render](https://render.com/).

### Deployment Configuration

* **Build Command:** `pip install -r api/requirements.txt`
* **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
* **Python Version:** `3.11.9` (specified in `runtime.txt`)
* **Root Directory:** `/feedback_webapp` (or adjust based on your Render setup)

### Environment Variables (Optional)

You can configure these environment variables in Render:
* `MODEL_PATH`: Path to the model file (default: `subcategory_model_augmented.keras`)
* `SUB_CLASSES_PATH`: Path to the classes JSON file (default: `subcategory_classes.json`)

## CORS Configuration

The API is configured with permissive CORS settings (`allow_origins=["*"]`) to allow requests from any frontend. For production deployments, you may want to restrict this to specific domains.

## Model Features

The model uses two types of inputs:

1. **Text Input:** The raw feedback comment (preprocessed and cleaned)
2. **Keyword Features:** Four binary features indicating the presence of:
   - Beverage-related words
   - Service item-related words
   - Catering-related words
   - Crew-related words

These features help improve classification accuracy by providing domain-specific context to the BERT model.

## Support

For issues or questions, please contact [your contact information].
