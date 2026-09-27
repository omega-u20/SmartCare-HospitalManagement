# <a href="https://colab.research.google.com/github/omega-u20/SmartCare-HospitalManagement/blob/main/Task2_Dataset_Understanding.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Task 02 – Dataset Understanding

# Connecting to Google Drive

# from google.colab import drive
# drive.mount('/content/drive')

import os
DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))

# ## 1. Setup

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

df = pd.read_csv(f'{DATA_FOLDER}/smartcare_ai_dataset_1000.csv')
data_dict = pd.read_csv(f'{DATA_FOLDER}/smartcare_ai_dataset_data_dictionary.csv')

print('Dataset shape:', df.shape)
df.head()

# ## 2. Dataset Overview
# 
# The dataset contains **1000 patient records** across **33 columns**. Each row represents one hospital appointment/visit, combining patient demographics, clinical measurements, hospital operations and billing information.
# 
# The dataset actually supports three different prediction problems (no-show, readmission, disease risk) so it includes three target columns. For this coursework we are only using `readmitted_30_days` as the target.

print('Number of rows:', df.shape[0])
print('Number of columns:', df.shape[1])
print()
print('Column data types:')
print(df.dtypes.value_counts())

df.info()

# **Observation:** All columns are either whole numbers (`int64`), decimals (`float64`), or text (`object`). There is no obvious sign of corrupted formatting at this stage . This will be checked more carefully in the Data Quality section below.

taset overview

# ## 3. Attribute Description
# 
# The 33 columns can be grouped into five categories:
# 
# | Category | Columns |
# |---|---|
# | **Identifiers** | record_id, patient_id |
# | **Patient Demographics** | age, gender, blood_group |
# | **Appointment / Operations** | department, diagnosis, appointment_date, waiting_days, previous_appointments, missed_previous_appointments, appointment_status, admitted, room_type, length_of_stay_days, previous_admissions |
# | **Clinical Measurements** | systolic_bp, diastolic_bp, blood_sugar_mg_dl, cholesterol_mg_dl, bmi, lab_tests_count, treatments_count |
# | **Financial** | consultation_fee_lkr, room_charge_lkr, lab_charge_lkr, medicine_charge_lkr, total_bill_lkr, payment_status, payment_method |
# | **Target Variables** | no_show, readmitted_30_days, disease_risk_level |
# 
# Let's look at basic statistics for the numeric columns and the unique values for the categorical (text) columns.

# Summary statistics for numeric columns
df.describe()

# Categorical columns and their unique values
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('patient_id')  # identifier, not a real category
categorical_cols.remove('appointment_date')  # date, not a category

for col in categorical_cols:
    print(f"--- {col} ---")
    print(df[col].value_counts())
    print()

# ### Key attribute notes
# 
# - **age** ranges roughly across the adult population, no negative or impossible values expected , checked in Data Quality below.
# - **gender** and **blood_group** are simple categorical fields.
# - **department / diagnosis** describe what the visit was for.
# - **admitted** is a 0/1 flag. `room_type` and `length_of_stay_days` only make sense when `admitted = 1`, which is why we expect a lot of missing/None values in `room_type` for non-admitted patients.
# - **previous_admissions** and **missed_previous_appointments** are historical counts that are likely to be strong predictors of readmission, since past behaviour tends to predict future behaviour.
# - **systolic_bp, diastolic_bp, blood_sugar_mg_dl, cholesterol_mg_dl, bmi** are the core clinical vitals used to judge a patient's health condition.
# - **total_bill_lkr** is effectively the sum of consultation, room, lab, and medicine charges . This will be worth checking for consistency.

# ## 4. Target Variable – `readmitted_30_days`
# 
# This is the column we are trying to predict for Option B. It is a **binary classification** target:
# - `1` = patient was readmitted within 30 days
# - `0` = patient was not readmitted within 30 days

counts = df['readmitted_30_days'].value_counts()
percentages = df['readmitted_30_days'].value_counts(normalize=True) * 100

print('Class counts:')
print(counts)
print()
print('Class percentages:')
print(percentages.round(1))

# **Observation:** The dataset has 747 patients not readmitted (74.7%) and 253 patients readmitted (25.3%). This is a **moderately imbalanced** target  not extreme. But the minority class (readmitted) is about a third the size of the majority class. This is something we will need to handle carefully at the modelling stage (Task 05), for example by using class weighting, resampling or choosing evaluation metrics like F1-score and ROC-AUC instead of relying only on accuracy.

# ## 5. Input Variables vs Target Variable
# 
# For the readmission prediction task:
# 
# - **Target (label):** `readmitted_30_days`
# - **Candidate input features:** all other columns *except* the identifiers (`record_id`, `patient_id`) and the two unused target columns (`no_show`, `disease_risk_level`) which must be dropped so the model doesn't accidentally "cheat" by learning from a related target.
# 
# `appointment_date` is a raw date string and will need to be converted into something more useful (e.g. month, day of week) or dropped during Feature Engineering in Task 03  it cannot be fed into a model as is.

target = 'readmitted_30_days'
cols_to_exclude = ['record_id', 'patient_id', 'no_show', 'disease_risk_level', target]
candidate_features = [c for c in df.columns if c not in cols_to_exclude]

print('Target variable:', target)
print()
print(f'Candidate input features ({len(candidate_features)}):')
print(candidate_features)

# ## 6. Data Dictionary Interpretation
# 
# The dataset comes with an official data dictionary file (`smartcare_ai_dataset_data_dictionary.csv`) describing what every column means. Below we display it in full and highlight the columns most relevant to our readmission prediction task.

data_dict

# Highlight the dictionary rows most relevant to readmission prediction
readmission_relevant = ['previous_admissions', 'missed_previous_appointments', 'admitted',
                          'length_of_stay_days', 'room_type', 'systolic_bp', 'diastolic_bp',
                          'blood_sugar_mg_dl', 'cholesterol_mg_dl', 'bmi', 'age',
                          'lab_tests_count', 'treatments_count', 'readmitted_30_days']

data_dict[data_dict['Column'].isin(readmission_relevant)]

# **Interpretation:** These columns describe a patient's clinical severity (blood pressure, sugar, cholesterol, BMI) and their hospital history (previous admissions, length of stay, missed appointments). Together they form the most medically sensible set of predictors for whether someone comes back within 30 days . This matches how readmission risk is generally understood in healthcare. Sicker patients and patients with a history of frequent hospital contact are more likely to be readmitted.

# ## 7. Data Quality Check
# 
# A quick first look at data quality

# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0]
print('Columns with missing values:')
print(missing)
print()
print(f"Percentage of rows missing 'room_type': {df['room_type'].isnull().mean()*100:.1f}%")

# Cross-check: is room_type missing exactly when the patient was NOT admitted?
pd.crosstab(df['admitted'], df['room_type'].isnull(), rownames=['admitted'], colnames=['room_type is missing'])

# **Observation:** `room_type` is missing for 906 out of 1000 records (90.6%). This is **not a data error** . The crosstab confirms it is missing exactly for patients where `admitted = 0`. In other words, if a patient wasn't admitted, they were never assigned a room .So `room_type` is naturally blank. This will be handled in Task 03 by filling these with a value such as `'Not Admitted'` rather than treating it as a missing-data problem.

# Duplicate records
print('Fully duplicate rows:', df.duplicated().sum())
print('Duplicate patient IDs:', df['patient_id'].duplicated().sum())

# **Observation:** No duplicate rows and no duplicate patient IDs were found, so each record represents a unique patient visit.

# Sanity check on numeric ranges
df[['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'blood_sugar_mg_dl', 'cholesterol_mg_dl']].describe()

# **Observation:** The min/max values for age, BMI, and the clinical vitals look within medically plausible ranges (no negative ages, no zero BMI, etc.).

