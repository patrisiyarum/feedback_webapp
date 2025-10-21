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
