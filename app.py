import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(page_title="SmartCare Readmission Prediction", layout="wide")

# Injecting Medlio Custom CSS
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Typography and App Background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important; /* Deep, professional hospital blue */
        color: white !important;
    }
    
    /* Make top header transparent to show buttons */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Main container padding */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    /* Headings outside the form */
    h1, h2, h3, p {
        color: #ffffff !important;
    }
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }
    
    /* Keep headings inside the form dark */
    div[data-testid="stForm"] h1, 
    div[data-testid="stForm"] h2, 
    div[data-testid="stForm"] h3,
    div[data-testid="stForm"] p {
        color: #1a1a1a !important;
    }
    
    /* Glassmorphism Cards and Form Containers */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.35) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 24px !important;
        padding: 2rem !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07) !important;
    }

    /* Input Fields */
    .stNumberInput input, .stSelectbox > div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        border-radius: 12px !important;
        padding: 0.5rem 1rem;
        font-size: 0.95rem;
        color: #374151;
        transition: all 0.2s ease-in-out;
    }
    
    .stNumberInput input:focus, .stSelectbox > div[data-baseweb="select"]:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        background-color: #ffffff;
    }

    /* Input Labels */
    .stNumberInput label, .stSelectbox label {
        color: #4b5563 !important;
        font-weight: 500 !important;
        margin-bottom: 0.25rem;
    }

    /* Primary Submit Button (Hospital Blue Highlight) */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.35);
        transition: all 0.2s ease-in-out;
        width: 100%;
        margin-top: 1rem;
    }
    
    div[data-testid="stFormSubmitButton"] > button p {
        color: white !important;
        font-weight: 700 !important;
        letter-spacing: 0.025em;
    }
    
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%);
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.5);
        color: white !important;
    }

    /* Alert / Status Messages */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Horizontal Dividers */
    hr {
        border-top: 1px solid #e5e7eb;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Define path to artifacts - using a relative path now
MODEL_FOLDER = 'artifacts/models'

@st.cache_resource
def load_artifacts():
    model = joblib.load(f'{MODEL_FOLDER}/random_forest.joblib')
    scaler = joblib.load(f'{MODEL_FOLDER}/scaler.joblib')
    feature_cols = joblib.load(f'{MODEL_FOLDER}/feature_columns.joblib')
    numeric_cols = joblib.load(f'{MODEL_FOLDER}/numeric_columns.joblib')
    return model, scaler, feature_cols, numeric_cols

try:
    model, scaler, feature_cols, numeric_cols = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

st.title("SmartCare Hospital: AI Decision Support")
st.subheader("30-Day Patient Readmission Prediction & Analysis")
st.markdown("Enter patient details below to generate a readmission risk prediction and a comprehensive AI reasoning report.")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        admitted = st.selectbox("Admitted During Visit? (0=No, 1=Yes)", [0, 1])
        length_of_stay = st.number_input("Length of Stay (Days)", min_value=0, value=1)
        total_bill = st.number_input("Total Bill (LKR)", min_value=0.0, value=15000.0)

    with col2:
        treatments_count = st.number_input("Number of Treatments", min_value=0, value=2)
        medicine_charge = st.number_input("Medicine Charge (LKR)", min_value=0.0, value=5000.0)
        lab_tests_count = st.number_input("Lab Tests Count", min_value=0, value=1)
        lab_charge = st.number_input("Lab Charge (LKR)", min_value=0.0, value=2000.0)

    submit_button = st.form_submit_button("Generate Prediction & Report")

if submit_button:
    # Initialize an empty dataframe with all required features set to 0
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)

    # Populate with user inputs
    input_df['age'] = age
    input_df['admitted'] = admitted
    input_df['length_of_stay_days'] = length_of_stay
    input_df['total_bill_lkr'] = total_bill
    input_df['treatments_count'] = treatments_count
    input_df['medicine_charge_lkr'] = medicine_charge
    input_df['lab_tests_count'] = lab_tests_count
    input_df['lab_charge_lkr'] = lab_charge

    # Calculate engineered features exactly as done in preprocessing
    input_df['avg_charge_per_treatment'] = total_bill / treatments_count if treatments_count > 0 else total_bill

    # Scale numeric features
    input_df_scaled = input_df.copy()
    input_df_scaled[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Generate Prediction
    prediction = model.predict(input_df_scaled)[0]
    probability = model.predict_proba(input_df_scaled)[0][1]

    st.markdown("---")

    # 1. Display Primary Verdict
    st.subheader("1. Clinical Verdict")
    if prediction == 1:
        st.error(f"**High Risk of Readmission** (Probability: {probability:.1%})")
        st.warning("**Recommendation:** Flag for intensive clinical follow-up, medication reconciliation, and discharge review.")
    else:
        st.success(f"**Low Risk of Readmission** (Probability: {probability:.1%})")
        st.info("**Recommendation:** Proceed with standard discharge protocols.")

    # 2. Display Comprehensive SHAP Report
    st.markdown("---")
    st.subheader("2. Comprehensive AI Decision Report")
    st.markdown("This section breaks down exactly *why* the AI made this prediction. It shows how much each specific patient attribute pushed the risk score up or down.")

    with st.spinner("Generating SHAP Explainability Report..."):
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df_scaled)

        # Handle SHAP value outputs depending on sklearn/shap version
        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1]
        elif len(np.shape(shap_values)) == 3:
            shap_values_class1 = shap_values[:, :, 1]
        else:
            shap_values_class1 = shap_values

        expected_value = explainer.expected_value
        expected_value_class1 = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) and len(np.shape(expected_value)) > 0 and np.shape(expected_value)[0] > 1 else expected_value

        # Render the Waterfall Plot
        fig = plt.figure(figsize=(10, 6))
        shap.plots._waterfall.waterfall_legacy(
            expected_value_class1,
            shap_values_class1[0],
            input_df_scaled.iloc[0],
            max_display=10,
            show=False
        )
        plt.title('Feature Contributions to Readmission Risk')
        plt.tight_layout()

        # Show plot in Streamlit
        st.pyplot(fig)

    st.markdown("""
    **How to interpret this clinical chart:**
    * **E[f(x)] (Bottom):** The baseline expected risk for an average patient in the hospital.
    * **f(x) (Top):** The final calculated risk score for *this specific patient* before applying the final cutoff.
    * **Red Bars:** Patient factors that *increased* the risk of readmission.
    * **Blue Bars:** Patient factors that *decreased* the risk of readmission.
    * **Size of the bar:** Indicates the strength of that specific factor's impact on the final decision.
    """)
