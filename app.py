
import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessing files
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.title("Heart Disease Prediction")
st.markdown("Enter the patient's details to get an ML-based prediction.")

# User inputs
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("Sex", ["Male", "Female"])

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina",
     "Non-Anginal Pain", "Asymptomatic"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure", 80, 200, 120
)

cholesterol = st.number_input(
    "Cholesterol", 100, 600, 200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST-T Wave Abnormality",
     "Left Ventricular Hypertrophy"]
)

max_hr = st.number_input(
    "Maximum Heart Rate", 60, 220, 150
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina", ["Yes", "No"]
)

oldpeak = st.number_input(
    "Oldpeak", 0.0, 10.0, 1.0
)

st_slope = st.selectbox(
    "ST Slope", ["Upsloping", "Flat", "Downsloping"]
)

# Prediction
if st.button("Predict"):

    # Convert user inputs into model-compatible columns
    input_values = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "Yes" else 0,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        "Sex_M": 1 if sex == "Male" else 0,

        "ChestPainType_ATA": 1 if chest_pain == "Atypical Angina" else 0,
        "ChestPainType_NAP": 1 if chest_pain == "Non-Anginal Pain" else 0,
        "ChestPainType_TA": 1 if chest_pain == "Typical Angina" else 0,

        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST-T Wave Abnormality" else 0,

        "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,

        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Upsloping" else 0
    }

    # Create DataFrame in the exact training-column order
    input_data = pd.DataFrame([input_values])
    input_data = input_data.reindex(
        columns=expected_columns,
        fill_value=0
    )

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    # Find probability corresponding to class 1
    class_index = list(model.classes_).index(1)
    heart_disease_probability = probability[0][class_index]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Model predicts a higher likelihood of heart disease.")
    else:
        st.success("Model predicts a lower likelihood of heart disease.")

    st.write(
        f"Model probability for class 1: "
        f"{heart_disease_probability * 100:.2f}%"
    )

    st.caption(
        "Educational ML output only. This is not a medical diagnosis."
    )
