import streamlit as st
import pandas as pd
import joblib
st.set_page_config(

    page_title="Heart Disease Prediction",

    page_icon="❤️",

    layout="wide"
)
pipeline = joblib.load(

    "heart_disease_pipeline.pkl"

)
st.title("❤️ Heart Disease Risk Prediction")

st.write(
    "Predict the risk of heart disease using Machine Learning."
)
age = st.number_input(
    "Age",
    18,
    100,
    40
)

sex = st.selectbox(
    "Sex",
    ["M","F"]
)

cp = st.selectbox(
    "Chest Pain Type",
    ["ATA","NAP","ASY","TA"]
)

restingbp = st.number_input(
    "Resting Blood Pressure",
    80,
    220,
    120
)

cholesterol = st.number_input(
    "Cholesterol",
    0,
    700,
    200
)

fastingbs = st.selectbox(
    "Fasting Blood Sugar",
    [0,1]
)

restecg = st.selectbox(
    "Resting ECG",
    ["Normal","ST","LVH"]
)

maxhr = st.number_input(
    "Maximum Heart Rate",
    60,
    220,
    150
)

exerciseangina = st.selectbox(
    "Exercise Angina",
    ["N","Y"]
)

oldpeak = st.number_input(
    "Old Peak",
    0.0,
    10.0,
    1.0
)

stslope = st.selectbox(
    "ST Slope",
    ["Up","Flat","Down"]
)
if st.button("🔍 Predict Risk"):
        patient = pd.DataFrame({

        "Age":[age],

        "Sex":[sex],

        "ChestPainType":[cp],

        "RestingBP":[restingbp],

        "Cholesterol":[cholesterol],

        "FastingBS":[fastingbs],

        "RestingECG":[restecg],

        "MaxHR":[maxhr],

        "ExerciseAngina":[exerciseangina],

        "Oldpeak":[oldpeak],

        "ST_Slope":[stslope]

    })
prediction = pipeline.predict(patient)[0]

probability = pipeline.predict_proba(patient)[0]
confidence = round(

        max(probability)*100,

        2

    )
if prediction == 1:

        st.error("🔴 High Risk of Heart Disease")

        st.metric(
            "Confidence",
            f"{confidence}%"
        )

        st.warning(

            "Please consult a cardiologist."

        )

else:

        st.success("🟢 Low Risk")

        st.metric(
            "Confidence",
            f"{confidence}%"
        )

        st.info(

            "Maintain a healthy lifestyle."

        )
st.markdown("---")

st.caption(

    "⚠️ This application is for educational purposes only and should not be used as a substitute for professional medical advice."

)
st.metric(
    label="Prediction Confidence",
    value=f"{confidence:.2f}%"
)
st.subheader("Prediction Probability")

st.write(f"🟢 Low Risk : {probability[0]*100:.2f}%")

st.write(f"🔴 High Risk : {probability[1]*100:.2f}%")
if probability[1] < 0.40:

    risk = "🟢 LOW"

elif probability[1] < 0.70:

    risk = "🟡 MEDIUM"

else:

    risk = "🔴 HIGH"

st.subheader("Risk Level")

st.write(risk)