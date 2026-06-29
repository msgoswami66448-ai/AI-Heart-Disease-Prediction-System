
# ==========================================================
# Feature-1 : Import Required Libraries
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
# ==========================================================
# Feature-2 : Configure Streamlit Application
# ==========================================================

st.set_page_config(
    page_title="AI Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ==========================================================
# Feature-3 : Load Trained Machine Learning Pipeline
# ==========================================================

pipeline = joblib.load("heart_disease_pipeline.pkl")
# ==========================================================
# Feature-4 : Professional Header
# ==========================================================

st.title("❤️ AI-Powered Heart Disease Prediction System")

st.markdown(
"""
### Predict Heart Disease Risk Using Machine Learning

This application estimates the likelihood of heart disease using a trained
**Extra Trees Classifier** based on patient clinical information.
"""
)

st.divider()
# ==========================================================
# Feature-5 : Project Information
# ==========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="🧠 Model",
        value="Extra Trees"
    )

with col2:
    st.metric(
        label="📊 Accuracy",
        value="91.30%"
    )

with col3:
    st.metric(
        label="🗂 Dataset",
        value="918 Records"
    )

st.divider()
# ==========================================================
# Feature-6 : Professional Sidebar
# ==========================================================

with st.sidebar:

    st.header("❤️ About This Project")

    st.divider()

    st.subheader("🤖 Model")
    st.write("Extra Trees Classifier")

    st.subheader("📊 Accuracy")
    st.write("91.30 %")

    st.subheader("🗂 Dataset")
    st.write("918 Patient Records")

    st.divider()

    st.subheader("⚠️ Medical Disclaimer")

    st.info(
        """
        This application is developed for educational
        and research purposes only.

        It should not be used as a substitute for
        professional medical advice.
        """
    )

    st.divider()

    st.subheader("🔄 Version")

    st.success("Version 1.0.0")
    # ==========================================================
# Feature-7 : Dashboard Layout
# ==========================================================

left_column, right_column = st.columns([2, 1])
# ==========================================================
# Feature-8 : Left Panel
# ==========================================================

with left_column:

    st.subheader("👤 Patient Information")
    st.write("Patient clinical information form will be displayed here.")
    # ==========================================================
# Feature-9 : Right Panel
# ==========================================================

with right_column:

    st.subheader("📊 Prediction Dashboard")

    st.info(
        """
        Prediction results will appear here
        after clicking the **Predict Risk** button.
        """
    )


# ==========================================================
# Patient Basic Information
# ==========================================================


with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=40
            )

            cp = st.selectbox(
                "Chest Pain Type",
                ["ATA", "NAP", "ASY", "TA"]
            )

            cholesterol = st.number_input(
                "Cholesterol",
                min_value=0,
                max_value=700,
                value=200
            )

            restecg = st.selectbox(
                "Resting ECG",
                ["Normal", "ST", "LVH"]
            )

            exerciseangina = st.selectbox(
                "Exercise Angina",
                ["N", "Y"]
            )
# ==========================================================
# Patient Clinical Information
# ==========================================================

with col2:

            sex = st.selectbox(
                "Gender",
                ["M", "F"]
            )

            restingbp = st.number_input(
                "Resting Blood Pressure",
                min_value=80,
                max_value=220,
                value=120
            )

            fastingbs = st.selectbox(
                "Fasting Blood Sugar",
                [0, 1]
            )

            maxhr = st.number_input(
                "Maximum Heart Rate",
                min_value=60,
                max_value=220,
                value=150
            )

            oldpeak = st.number_input(
                "Old Peak",
                min_value=0.0,
                max_value=10.0,
                value=1.0
            )

            stslope = st.selectbox(
                "ST Slope",
                ["Up", "Flat", "Down"]
            )
# ==========================================================
# Submit Button
# ==========================================================

predict_button = st.form_submit_button(
            "🔍 Predict Risk",
            use_container_width=True
        )
# ==========================================================
# Feature-11 : Execute Prediction
# ==========================================================

if predict_button:
    # ==========================================================
# Create Patient DataFrame
# ==========================================================

    patient = pd.DataFrame({

        "Age": [age],

        "Sex": [sex],

        "ChestPainType": [cp],

        "RestingBP": [restingbp],

        "Cholesterol": [cholesterol],

        "FastingBS": [fastingbs],

        "RestingECG": [restecg],

        "MaxHR": [maxhr],

        "ExerciseAngina": [exerciseangina],

        "Oldpeak": [oldpeak],

        "ST_Slope": [stslope]

    })# ==========================================================
# Predict Disease Risk
# ==========================================================

    prediction = pipeline.predict(patient)[0]

    probability = pipeline.predict_proba(patient)[0]
    # ==========================================================
# Calculate Confidence
# ==========================================================

    confidence = round(

        max(probability) * 100,

        2

    )
    # ==========================================================
# Temporary Testing
# ==========================================================

    st.success("Prediction Completed Successfully")

    st.write("Prediction :", prediction)

    st.write("Confidence :", confidence)

    st.write("Probability :", probability)