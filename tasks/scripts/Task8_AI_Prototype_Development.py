# <a href="https://colab.research.google.com/github/omega-u20/SmartCare-HospitalManagement/blob/main/Task8_AI_Prototype_Development.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Attach google drive

# from google.colab import drive
# drive.mount('/content/drive')

import os
DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))
MODEL_FOLDER = f'{DATA_FOLDER}/models'

# # Install streamlit and localtunnel

!pip install streamlit shap matplotlib -q
!npm install localtunnel -q

# # Create streamlit python script

%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="SmartCare Readmission Prediction", layout="wide")

# Define path to artifacts
MODEL_FOLDER = '/content/drive/MyDrive/Group 16_CCS3440 - Asg 2/artifacts/models'

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

# # Run Streamlit App

import time
# from google.colab import output

# 1. Kill any stuck background processes from previous attempts
!pkill -f streamlit
!pkill -f localtunnel

# 2. Run Streamlit in the background with CORS and XSRF protection disabled
!nohup streamlit run app.py --server.port 8501 --server.enableCORS=false --server.enableXsrfProtection=false &> nohup.out &

# Wait a few seconds for the Streamlit server to boot up
time.sleep(3)

# 3. Embed the app securely as an iframe right inside the notebook
output.serve_kernel_port_as_iframe(8501, height=1080)

