# <a href="https://colab.research.google.com/github/omega-u20/SmartCare-HospitalManagement/blob/main/Task3_Preprocessing_Feature_Engineering.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Task 03 – Data Preprocessing and Feature Engineering

# Connecting to google drive

# from google.colab import drive
# drive.mount('/content/drive')

import os
DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))

# ## 1. Setup

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

pd.set_option('display.max_columns', None)
df = pd.read_csv(f'{DATA_FOLDER}/smartcare_ai_dataset_1000.csv')

print('Original shape:', df.shape)
df.head()


# ## 2. Missing Value Handling
# 
# From Task 02, we found only one column with missing values: `room_type`, missing in 906 out of 1000 rows (90.6%). We confirmed this is **not random**  it is missing exactly when `admitted = 0`, because a patient who wasn't admitted was never assigned a room.
# 
# **Decision:** We do not drop this column (it still carries useful information for admitted patients)and we do not use mean/mode imputation (which would invent a fake room type). Instead we fill missing values with a new category `'Not Admitted'` which is medically accurate.

print('Missing values before:')
print(df.isnull().sum()[df.isnull().sum() > 0])

df['room_type'] = df['room_type'].fillna('Not Admitted')

print()
print('Missing values after:')
print(df.isnull().sum().sum(), 'total missing values remaining')

# ## 3. Duplicate Record Detection
# 
# **Decision:** Check for fully duplicated rows and duplicated `patient_id` values. Task 02 already showed there were none. But we re-verify here as part of the formal cleaning pipeline (good practice, and safer if the dataset changes).

print('Fully duplicate rows:', df.duplicated().sum())
print('Duplicate patient IDs:', df['patient_id'].duplicated().sum())

df = df.drop_duplicates()
print('Shape after duplicate removal:', df.shape)

# ## 4. Outlier Identification
# 
# **Method:** We use the IQR (Interquartile Range) method: any value below `Q1 - 1.5*IQR` or above `Q3 + 1.5*IQR` is flagged as a statistical outlier.
# 
# **Important distinction:** a statistical outlier is not automatically an *error*. We check whether flagged values are still medically/operationally plausible before deciding what to do with them.

def iqr_outlier_summary(data, columns):
    rows = []
    for col in columns:
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((data[col] < low) | (data[col] > high)).sum()
        rows.append({'column': col, 'min': data[col].min(), 'max': data[col].max(),
                      'lower_bound': round(low, 1), 'upper_bound': round(high, 1),
                      'num_outliers': n_out})
    return pd.DataFrame(rows).sort_values('num_outliers', ascending=False)

clinical_cols = ['age', 'systolic_bp', 'diastolic_bp', 'blood_sugar_mg_dl', 'cholesterol_mg_dl', 'bmi']
iqr_outlier_summary(df, clinical_cols)

# **Observation:** The clinical vitals (blood pressure, sugar, cholesterol, BMI) have very few outliers (single digits) and their min/max values are all within medically realistic ranges e.g BMI between 14.0 and 38.8 systolic BP between 85 and 178. These are real patient variation not data errors.
# 
# **Decision:** Do not remove any rows for these columns. Removing them would throw away genuine clinical signal that may be useful for predicting readmission (e.g very high blood sugar is a real risk factor not noise).

financial_cols = ['consultation_fee_lkr', 'room_charge_lkr', 'lab_charge_lkr', 'medicine_charge_lkr', 'total_bill_lkr']
iqr_outlier_summary(df, financial_cols)

# **Observation:** Billing columns show more flagged outliers, but this is expected  charges are naturally right-skewed (most visits are cheap, a handful of ICU/long-stay cases are very expensive). We also verified in Task 02 that `total_bill_lkr` always equals the exact sum of the four charge components, confirming the billing data is internally consistent and not corrupted.
# 
# **Decision:** Keep these values as-is. High bills are expected to correlate with admission severity and are potentially useful predictors, not noise to be removed.

history_cols = ['previous_appointments', 'missed_previous_appointments', 'previous_admissions',
                 'length_of_stay_days', 'lab_tests_count', 'treatments_count']
iqr_outlier_summary(df, history_cols)

# **Observation:** These are small integer counts (e.g `previous_admissions` ranges 0–5). The IQR method flags many of them simply because the distribution is tight and skewed toward zero not because the values are unrealistic. A patient with 5 previous admissions is unusual but entirely plausible and is exactly the kind of patient we most want the model to learn from for a readmission task.
# 
# **Decision:** Keep all values. No capping or removal these are precisely the high-risk cases the model needs to see.

# ## 5. Data Cleaning Summary
# 
# | Issue | Decision | Reason |
# |---|---|---|
# | `room_type` missing (90.6%) | Filled with `'Not Admitted'` | Missing is logical (no room if not admitted), not random |
# | Duplicate rows | Checked, none found | Confirms one row = one unique visit |
# | Outliers (clinical, financial, history) | Kept as-is | Values are plausible and often carry real predictive signal |
# | `total_bill_lkr` consistency | Verified against sum of charges | Confirms billing data integrity, no correction needed |

# ## 6. Feature Selection – Dropping Unneeded Columns
# 
# **Decision:** Remove columns that cannot or should not be used as model inputs:
# - `record_id`, `patient_id` - unique identifiers, carry no predictive information and could cause the model to memorise individual rows.
# - `no_show`, `disease_risk_level` - these are the target variables for the *other two* prediction options (A and C). Leaving them in would let the model "cheat" using information that is not realistically available at prediction time, and they are outside the scope of Option B.

df_model = df.drop(columns=['record_id', 'patient_id', 'no_show', 'disease_risk_level'])
print('Shape after dropping identifiers and unused targets:', df_model.shape)
df_model.columns.tolist()

# ## 7. Feature Engineering
# 
# We create a small number of new features that are more informative for a readmission model than the raw columns alone and convert the raw date into something usable.

# ### 7.1 Date feature extraction
# `appointment_date` is a raw date string (e.g. `2025-04-10`). A model cannot use text dates directly and the exact calendar date itself has little meaning for readmission risk. We extract the **month** and **day of week** which can capture seasonal or weekly patterns, then drop the original date column.

df_model['appointment_date'] = pd.to_datetime(df_model['appointment_date'])
df_model['appointment_month'] = df_model['appointment_date'].dt.month
df_model['appointment_dayofweek'] = df_model['appointment_date'].dt.dayofweek  # 0=Monday
df_model = df_model.drop(columns=['appointment_date'])

df_model[['appointment_month', 'appointment_dayofweek']].head()

# ### 7.2 Engineered risk features
# Based on the domain knowledge from Task 02 (patients with more hospital history and worse vitals are more likely to be readmitted)  we create a few combined features:
# 
# - **`missed_appointment_rate`** - proportion of previous appointments that were missed. This captures patient engagement better than a raw count (a patient with 2 missed out of 3 is very different from 2 missed out of 10).
# - **`is_hypertensive`** - flag for blood pressure in a commonly used clinical high range (systolic ≥ 140 or diastolic ≥ 90)turning raw numbers into a clinically meaningful signal.
# - **`avg_charge_per_treatment`** - total bill divided by number of treatments indicating cost intensity per treatment given.

# missed_appointment_rate — guard against divide-by-zero for patients with no previous appointments
df_model['missed_appointment_rate'] = np.where(
    df_model['previous_appointments'] > 0,
    df_model['missed_previous_appointments'] / df_model['previous_appointments'],
    0
)

# is_hypertensive flag
df_model['is_hypertensive'] = ((df_model['systolic_bp'] >= 140) | (df_model['diastolic_bp'] >= 90)).astype(int)

# avg_charge_per_treatment — guard against divide-by-zero
df_model['avg_charge_per_treatment'] = np.where(
    df_model['treatments_count'] > 0,
    df_model['total_bill_lkr'] / df_model['treatments_count'],
    df_model['total_bill_lkr']
)

df_model[['missed_appointment_rate', 'is_hypertensive', 'avg_charge_per_treatment']].describe()

# ## 8. Feature Encoding
# 
# The model needs numbers, not text. So all categorical columns must be encoded.
# 
# **Decision:**
# - **One-Hot Encoding** for columns with no natural order (nominal): `gender`, `blood_group`, `department`, `diagnosis`, `room_type`, `payment_status`, `payment_method`, `appointment_status`. Each category becomes its own 0/1 column which avoids implying a false ranking between categories.
# - We use `drop_first=True` to avoid redundant columns (dummy variable trap).

categorical_cols = ['gender', 'blood_group', 'department', 'diagnosis', 'room_type',
                     'payment_status', 'payment_method', 'appointment_status']

print('Cardinality of each categorical column:')
for col in categorical_cols:
    print(f'  {col}: {df_model[col].nunique()} categories')

df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
print()
print('Shape after one-hot encoding:', df_model.shape)

# ## 9. Feature Scaling
# 
# Numeric columns are on very different scales (e.g. `age` is 0–90, `total_bill_lkr` is in the thousands). Some algorithms (Logistic Regression, KNN, SVM) are sensitive to this and will let large-scale columns dominate unless we standardise.
# 
# **Decision:** Use `StandardScaler` (mean = 0, standard deviation = 1) on the numeric columns. Tree-based models (Decision Tree, Random Forest, XGBoost) don't need scaling but it does no harm to them so we scale once here and keep the scaled version for all models to keep the pipeline consistent.

numeric_cols = ['age', 'waiting_days', 'previous_appointments', 'missed_previous_appointments',
                 'admitted', 'length_of_stay_days', 'previous_admissions', 'systolic_bp', 'diastolic_bp',
                 'blood_sugar_mg_dl', 'cholesterol_mg_dl', 'bmi', 'lab_tests_count', 'treatments_count',
                 'consultation_fee_lkr', 'room_charge_lkr', 'lab_charge_lkr', 'medicine_charge_lkr',
                 'total_bill_lkr', 'appointment_month', 'appointment_dayofweek',
                 'missed_appointment_rate', 'is_hypertensive', 'avg_charge_per_treatment']

scaler = StandardScaler()
df_model[numeric_cols] = scaler.fit_transform(df_model[numeric_cols])

df_model[numeric_cols].describe().round(2).loc[['mean', 'std']]

# The scaler is fit here for demonstration.  This copy is kept for the EDA and feature-selection steps below.

# ## 10. Feature Selection – Correlation Check
# 
# Before finalising features, we check correlation of numeric features with the target to see which ones look most useful and check for redundant features that are highly correlated with each other.

correlations = df_model[numeric_cols + ['readmitted_30_days']].corr()['readmitted_30_days'].drop('readmitted_30_days')
correlations.sort_values(ascending=False)

# **Observation:** As expected from our domain reasoning in Task 02, history-related features (`previous_admissions`, `missed_previous_appointments`) and clinical severity features tend to show the strongest relationship with `readmitted_30_days`. None of the individual correlations are extremely high on their own .This suggests readmission is driven by a *combination* of factors rather than any single variable which supports using models capable of capturing interactions (Random Forest, XGBoost) rather than relying on one or two features.

# Check for highly correlated (redundant) feature pairs
corr_matrix = df_model[numeric_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
redundant_pairs = [(col, row, upper.loc[row, col]) for col in upper.columns for row in upper.index
                    if upper.loc[row, col] > 0.8]
print('Highly correlated pairs (>0.8):')
for pair in redundant_pairs:
    print(pair)

# **Observation:** The check found three redundant pairs (correlation > 0.8):
# - `length_of_stay_days` & `admitted` (0.83) - expected, since a stay of 0 days almost always means the patient wasn't admitted.
# - `lab_charge_lkr` & `lab_tests_count` (0.88) - expected, since lab charges are driven directly by how many tests were run.
# - `total_bill_lkr` & `room_charge_lkr` (0.88) - expected, since `total_bill_lkr` is partly built from `room_charge_lkr`.
# 
# **Decision:** We keep all of these features for now rather than dropping any, because tree-based models (Random Forest, XGBoost) handle correlated features without much issue and removing them risks losing information. This is flagged here so it can be revisited in Task 05 if a linear model (e.g. Logistic Regression) shows instability in that case, one column from each redundant pair could be dropped.

# ## 11. Final Preprocessed Dataset

print('Final shape:', df_model.shape)
print('Target column present:', 'readmitted_30_days' in df_model.columns)
df_model.head()

# Save the cleaned, preprocessed dataset for use in Task 04 (EDA) and Task 05 (Modelling)
df_model.to_csv(f'{DATA_FOLDER}/smartcare_preprocessed_readmission.csv', index=False)
print('Saved: smartcare_preprocessed_readmission.csv')

