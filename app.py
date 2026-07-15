# """
Stroke Risk Predictor — Streamlit App
Trained ML model deployed live on Streamlit Cloud

This app:
1. Loads data from a local CSV
2. Trains a model on startup (cached)
3. Makes predictions based on user input

Deploy: Push to GitHub, connect to Streamlit Cloud
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib

# ================================================================
# Page Config
# ================================================================
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ================================================================
# Load & Cache Data
# ================================================================
DATA_PATH = "data/healthcare-dataset-stroke-data.csv"


@st.cache_data
def load_data(path):
    """Load the dataset. Cached so it only runs once."""
    if not os.path.exists(path):
        return None, "not_found"
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, "parse_error"
    if df.shape[1] == 1:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            pass
    df.columns = [c.strip() for c in df.columns]
    return df, "ok"


# ================================================================
# Train Model (cached, runs once per session)
# ================================================================
@st.cache_resource
def train_model(df):
    """Train the stroke prediction model"""
    try:
        # Prepare data
        df_clean = df.dropna(subset=['bmi', 'age', 'avg_glucose_level'])
        df_clean['bmi'] = pd.to_numeric(df_clean['bmi'], errors='coerce')
        df_clean = df_clean.dropna(subset=['bmi'])

        # Encode categorical features
        le_gender = LabelEncoder()
        le_work = LabelEncoder()
        le_residence = LabelEncoder()
        le_smoking = LabelEncoder()
        le_married = LabelEncoder()

        df_encoded = df_clean.copy()
        df_encoded['gender'] = le_gender.fit_transform(df_clean['gender'])
        df_encoded['work_type'] = le_work.fit_transform(df_clean['work_type'])
        df_encoded['Residence_type'] = le_residence.fit_transform(df_clean['Residence_type'])
        df_encoded['smoking_status'] = le_smoking.fit_transform(df_clean['smoking_status'])
        df_encoded['ever_married'] = le_married.fit_transform(df_clean['ever_married'])

        # Features and target
        X = df_encoded[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
        y = df_encoded['stroke']

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # Train Random Forest (no SMOTE to avoid version conflicts)
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced')
        model.fit(X_imputed, y)

        # Store encoders for later use
        encoders = {
            'gender': le_gender,
            'work_type': le_work,
            'residence': le_residence,
            'smoking': le_smoking,
            'married': le_married,
            'imputer': imputer
        }

        return model, encoders

    except Exception as e:
        st.error(f"❌ Model training error: {str(e)}")
        return None, None

# ================================================================
# Load Data & Train Model
# ================================================================
df, status = load_data(DATA_PATH)

if status == "not_found":
    st.error(f"❌ CSV file not found at `{DATA_PATH}`. Make sure it's committed to the repo.")
    st.stop()
elif status == "parse_error":
    st.error(f"❌ Couldn't parse `{DATA_PATH}` as CSV. Check the file isn't corrupted or empty.")
    st.stop()

model, encoders = train_model(df)
if model is None or encoders is None:
    st.stop()

# ================================================================
# UI Header
# ================================================================
st.title("🧠 Stroke Risk Predictor")
st.caption("ML-powered health screening tool — Educational demo only")
st.warning(
    "⚠️ **Disclaimer:** This is an educational portfolio project, **NOT** a medical "
    "diagnosis tool. Always consult a qualified healthcare professional for medical concerns."
)

st.divider()

# ================================================================
# Input Form
# ================================================================
st.subheader("Enter Patient Information")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", min_value=0, max_value=100, value=45, step=1)
        hypertension = st.radio("Hypertension?", ["No", "Yes"], horizontal=True)
        heart_disease = st.radio("Heart disease?", ["No", "Yes"], horizontal=True)
        ever_married = st.radio("Ever married?", ["No", "Yes"], horizontal=True)

    with col2:
        work_type = st.selectbox(
            "Work type",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
        )
        residence_type = st.selectbox("Residence type", ["Urban", "Rural"])
        avg_glucose = st.number_input(
            "Avg glucose level (mg/dL)",
            min_value=40.0, max_value=300.0, value=100.0, step=1.0
        )
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        smoking_status = st.selectbox(
            "Smoking status",
            ["never smoked", "formerly smoked", "smokes", "Unknown"]
        )

    submitted = st.form_submit_button("🔮 Predict Stroke Risk", use_container_width=True)

# ================================================================
# Make Prediction
# ================================================================
if submitted:
    try:
        # Encode categorical inputs
        gender_encoded = encoders['gender'].transform([gender])[0]
        work_encoded = encoders['work_type'].transform([work_type])[0]
        residence_encoded = encoders['residence'].transform([residence_type])[0]
        smoking_encoded = encoders['smoking'].transform([smoking_status])[0]
        married_encoded = encoders['married'].transform([ever_married])[0]

        # Create input array
        input_array = np.array([[
            gender_encoded,
            age,
            1 if hypertension == "Yes" else 0,
            1 if heart_disease == "Yes" else 0,
            married_encoded,
            work_encoded,
            residence_encoded,
            avg_glucose,
            bmi,
            smoking_encoded
        ]])

        # Impute missing values
        input_imputed = encoders['imputer'].transform(input_array)

        # Make prediction
        risk_score = model.predict_proba(input_imputed)[0][1] * 100
        prediction = model.predict(input_imputed)[0]

        st.divider()
        st.subheader("📊 Prediction Result")

        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Stroke Risk Score", f"{risk_score:.1f}%")
        with col2:
            risk_level = "🔴 High Risk" if risk_score > 50 else "🟡 Medium Risk" if risk_score > 25 else "🟢 Low Risk"
            st.metric("Risk Level", risk_level)

        # Display progress bar
        st.progress(min(int(risk_score), 100) / 100)

        # Display interpretation
        st.divider()
        if prediction == 1:
            st.error(
                "🔴 **Higher Risk Signal Detected**\n\n"
                "The model identifies this profile as having characteristics similar to "
                "stroke cases in the training data. This is not a diagnosis, but a signal "
                "to:\n"
                "- Schedule a check-up with your doctor\n"
                "- Monitor blood pressure regularly\n"
                "- Maintain a healthy lifestyle"
            )
        else:
            st.success(
                "🟢 **Lower Risk Signal**\n\n"
                "The model does not flag this profile as high-risk based on the input data. "
                "However:\n"
                "- Regular health check-ups are still recommended\n"
                "- Maintain a healthy lifestyle\n"
                "- This is not a medical guarantee"
            )

        # Show input data
        with st.expander("📋 View Input Data"):
            input_df = pd.DataFrame({
                "Feature": ["Gender", "Age", "Hypertension", "Heart Disease", "Ever Married",
                           "Work Type", "Residence", "Avg Glucose", "BMI", "Smoking Status"],
                "Value": [gender, age, hypertension, heart_disease, ever_married,
                         work_type, residence_type, avg_glucose, bmi, smoking_status]
            })
            st.dataframe(input_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        st.info("Please check your inputs and try again.")

st.divider()
st.caption(
    "Built with scikit-learn • Trained on Stroke Prediction Dataset "
    "[fedesoriano](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)"
)