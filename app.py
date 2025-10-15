import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Required for BERT ops
from keras.layers import TFSMLayer
import pandas as pd
import json
import numpy as np

# ============================================================
# ⚙️ Model + Class Loaders
# ============================================================
@st.cache_resource
def load_model_cached():
    model = TFSMLayer("two_layer_categorization_model_fixed", call_endpoint="serving_default")
    return model

@st.cache_resource
def load_classes_cached():
    with open("main_category_classes.json") as f:
        main_classes = json.load(f)
    with open("subcategory_classes.json") as f:
        sub_classes = json.load(f)
    return main_classes, sub_classes

try:
    model = load_model_cached()
    main_classes, sub_classes = load_classes_cached()
    st.toast("✅ Model loaded successfully!", icon="✅")
except Exception as e:
    st.error(f"❌ Failed to load model or class files: {e}")
    st.stop()

# ============================================================
# 🔮 Prediction Function
# ============================================================
def predict_comment(text_input_str):
    try:
        inputs = tf.constant([text_input_str])
        outputs = model(inputs)
        main_probs = outputs["main_category_output"][0].numpy() * 100
        sub_probs = outputs["subcategory_output"][0].numpy() * 100
        main_sorted = sorted(zip(main_classes, main_probs), key=lambda x: x[1], reverse=True)
        sub_sorted = sorted(zip(sub_classes, sub_probs), key=lambda x: x[1], reverse=True)
        return main_sorted, sub_sorted
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return [], []

# ============================================================
# 🎨 Styling
# ============================================================
def load_custom_css():
    st.markdown("""
    <style>
    :root {
        --bg: #111315;
        --card: #1b1d1f;
        --text: #f5f5f5;
        --muted: #c0c0c0;
        --accent: #4f8ff7;
        --border: #2a2c2e;
        --radius: 10px;
    }

    [data-testid="stAppViewContainer"] {
        background-color: var(--bg);
        color: var(--text);
    }

    .main-header {
        background: var(--card);
        border-radius: var(--radius);
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid var(--border);
    }

    .main-header h1 {
        font-size: 1.9rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
        color: var(--text);
    }
    .main-header p {
        color: var(--muted);
        font-size: 1rem;
    }

    .notion-card {
        background: var(--card);
        border-radius: var(--radius);
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border);
    }

    /* Buttons */
    .stButton > button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: var(--radius);
        padding: .6rem 1.4rem;
        font-weight: 500;
        transition: all .25s ease;
    }
    .stButton > button:hover {
        background: #3b6ee0;
        transform: translateY(-1px);
    }

    /* Expander styling */
    .stExpander {
        background-color: #1f2224;
        border: 1px solid #2a2c2e;
        border-radius: 10px;
        margin-bottom: 0.6rem;
    }

    .stExpander:hover {
        background-color: #25282b;
        transition: background 0.3s ease;
    }

    /* Text area */
    .stTextArea > div > div > textarea {
        background: #1c1e1f;
        color: var(--text);
        border-radius: var(--radius);
        border: 1px solid var(--border);
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 📊 Prediction Display
# ============================================================
def display_predictions(main_preds, sub_preds):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Main Category Predictions")
        for label, prob in main_preds[:3]:
            st.markdown(f"- **{label}** — {prob:.2f}%")
    with col2:
        st.markdown("### 📁 Subcategory Predictions")
        for label, prob in sub_preds[:3]:
            st.markdown(f"- **{label}** — {prob:.2f}%")

# ============================================================
# 🧭 Sidebar
# ============================================================
st.set_page_config(page_title="FCR Feedback Categorization", page_icon="🧠", layout="wide")
load_custom_css()

with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("", ["🏠 Home", "📊 Analytics", "ℹ️ About"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Model:** Fine-tuned BERT  \n**Output:** Main + Subcategory  \n**Version:** 2.2")

# ============================================================
# 🏠 HOME PAGE
# ============================================================
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>Model for FCR Feedback Categorization</h1>
        <p>AI-powered classification of feedback into structured main and subcategories.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="notion-card">', unsafe_allow_html=True)
    st.markdown("### 💬 Sample Comments by Category")

    samples = {
        "Food Safety": "The meal smelled funny and made me sick.",
        "Catering Error": "Required meals not catered.",
        "Food Quality": "The soup lacked flavor and the chicken was overcooked.",
        "Missing": "Insufficient beverages for expected load."
    }

    # Two-column expandable layout
    col1, col2 = st.columns(2)
    sample_items = list(samples.items())

    for i, (label, text) in enumerate(sample_items):
        col = col1 if i % 2 == 0 else col2
        with col.expander(label):
            st.markdown(f"**Example:** {text}")
            if st.button("Use this", key=f"use_{label}"):
                st.session_state.comment_text = text

    user_input = st.text_area(
        "Enter feedback comment:",
        placeholder="Example: 'The burger was juicy and perfectly cooked.'",
        height=120,
        key="comment_text"
    )

    if st.button("Analyze Feedback", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter a comment first.")
        else:
            with st.spinner("Analyzing..."):
                main_preds, sub_preds = predict_comment(user_input)
            if main_preds and sub_preds:
                st.success("Prediction complete!")
                display_predictions(main_preds, sub_preds)
            else:
                st.error("Prediction failed.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================
    # 📁 Bulk Upload for CSV Prediction
    # ========================================================
    st.markdown('<div class="notion-card">', unsafe_allow_html=True)
    st.markdown("### 📂 Bulk Upload for CSV Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.dataframe(df.head())

            text_col = next((c for c in df.columns if any(k in c.lower() for k in ["comment", "feedback", "review"])), None)
            if not text_col:
                st.error("Could not find a valid text column.")
            else:
                st.info(f"Detected text column: **{text_col}**")
                if st.button("Start Prediction", use_container_width=True):
                    progress = st.progress(0)
                    results = []
                    total = len(df)

                    for i, text in enumerate(df[text_col]):
                        main_preds, sub_preds = predict_comment(str(text))
                        if main_preds and sub_preds:
                            results.append({
                                "Predicted_Subcategory": sub_preds[0][0],
                                "Predicted_Main_Category": main_preds[0][0]
                            })
                        else:
                            results.append({"Predicted_Subcategory": "Error", "Predicted_Main_Category": "Error"})
                        progress.progress((i + 1) / total)

                    df["Predicted_Subcategory"] = [r["Predicted_Subcategory"] for r in results]
                    df["Predicted_Main_Category"] = [r["Predicted_Main_Category"] for r in results]
                    st.dataframe(df.head())

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Updated CSV", csv, "categorized_feedback.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 📊 ANALYTICS PAGE
# ============================================================
elif page == "📊 Analytics":
    st.markdown("""
    <div class="main-header">
        <h1>Analytics Dashboard</h1>
        <p>View category trends and feedback distributions.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("Analytics will appear after bulk uploads.")

# ============================================================
# ℹ️ ABOUT PAGE
# ============================================================
else:
    st.markdown("""
    <div class="main-header">
        <h1>About This Tool</h1>
        <p>Learn more about the FCR Feedback Categorization model.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    This tool uses a fine-tuned BERT-based model to classify feedback into **main** and **subcategory** labels,
    helping operations teams identify key trends quickly and accurately.  
    Built with Streamlit · TensorFlow · Keras.
    """)
