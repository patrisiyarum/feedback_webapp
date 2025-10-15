# app.py (This is now your Frontend)

import streamlit as st
import requests
import pandas as pd

# --- Configuration ---
# Set the title and icon of the page.
st.set_page_config(
    page_title="Feedback Categorization",
    page_icon="📊",
    layout="wide"
)

# This is the URL where your FastAPI backend is running.
# When running locally, it's usually http://127.0.0.1:8000
# When deployed, you'll replace this with your Render URL.
API_URL = "https://feedback-webapp-5zc2.onrender.com/predict"

# --- Page Title and Description ---
st.title("FCR Feedback Categorization Tool 📊")
st.markdown(
    "This tool uses a machine learning model to analyze and categorize customer feedback. "
    "Enter a single comment for instant analysis or upload a CSV file for batch processing."
)

# --- Main Application Logic ---

# We use st.tabs to create a clean, organized layout.
tab1, tab2 = st.tabs(["✍️ Single Comment Analysis", "📁 Batch Analysis (CSV)"])

# --- Tab 1: Single Comment Analysis ---
with tab1:
    st.header("Analyze a Single Piece of Feedback")
    
    # Text area for user input.
    user_text = st.text_area(
        "Enter the customer feedback text below:", 
        height=150, 
        placeholder="e.g., 'The flight attendant was very helpful when I needed to change my seat.'"
    )

    # Analyze button.
    if st.button("Analyze Comment"):
        if user_text and user_text.strip():
            with st.spinner("🧠 Analyzing... Please wait."):
                try:
                    # The data we send to the API must match the Pydantic model in main.py
                    payload = {"text": user_text}
                    
                    # Make the POST request to the API
                    response = requests.post(API_URL, json=payload)
                    
                    # Check if the request was successful
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Analysis Complete!")
                        
                        # Display the results in a more visual way
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                label="**Predicted Main Category**", 
                                value=result['main_category']
                            )
                            st.progress(result['main_category_confidence'])
                            st.write(f"Confidence: {result['main_category_confidence']:.2%}")
                        with col2:
                            st.metric(
                                label="**Predicted Sub-Category**", 
                                value=result['sub_category']
                            )
                            st.progress(result['sub_category_confidence'])
                            st.write(f"Confidence: {result['sub_category_confidence']:.2%}")

                    else:
                        # Display an error if the API call fails
                        st.error(f"Error from API: {response.status_code} - {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to the API. Please ensure the backend is running. Error: {e}")
        else:
            st.warning("Please enter some text to analyze.")

# --- Tab 2: Batch Analysis from CSV file ---
with tab2:
    st.header("Analyze Feedback from a CSV File")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV file with a 'text' column containing the feedback.", 
        type=["csv"]
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'text' not in df.columns:
                st.error("The CSV file must contain a column named 'text'.")
            else:
                st.write("**Preview of your data:**")
                st.dataframe(df.head())

                if st.button("Analyze Entire File"):
                    results_list = []
                    progress_bar = st.progress(0)
                    total_rows = len(df)

                    with st.spinner(f"Analyzing {total_rows} rows... This may take a moment."):
                        for i, row in df.iterrows():
                            # Send each row's text to the API
                            payload = {"text": row['text']}
                            response = requests.post(API_URL, json=payload)
                            
                            if response.status_code == 200:
                                results_list.append(response.json())
                            else:
                                # Append an error message if a row fails
                                results_list.append({
                                    "main_category": "Error", 
                                    "sub_category": f"API Error {response.status_code}",
                                    "main_category_confidence": 0.0,
                                    "sub_category_confidence": 0.0,
                                })

                            # Update the progress bar
                            progress_bar.progress((i + 1) / total_rows)

                    st.success("Batch analysis complete!")
                    
                    # Combine results with the original dataframe
                    results_df = pd.DataFrame(results_list)
                    final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
                    
                    st.write("**Analysis Results:**")
                    st.dataframe(final_df)

                    # Provide a download button for the results
                    csv = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name='feedback_analysis_results.csv',
                        mime='text/csv',
                    )

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
