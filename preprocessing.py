# ================================
# CUSTOMER CHURN PREDICTION SYSTEM
# Phase 3 - Data Cleaning & Feature Engineering
# ================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---- STEP 1: Load Data ----
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("✅ Data loaded:", df.shape)

# ---- STEP 2: Fix TotalCharges (string → number) ----
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("\n✅ TotalCharges fixed")
print("Missing values after fix:", df['TotalCharges'].isnull().sum())

# ---- STEP 3: Fill missing TotalCharges ----
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
print("✅ Missing values filled")

# ---- STEP 4: Drop customerID (useless for ML) ----
df.drop('customerID', axis=1, inplace=True)
print("✅ customerID dropped")

# ---- STEP 5: Encode Yes/No columns → 1/0 ----
binary_cols = ['Partner', 'Dependents', 'PhoneService',
               'PaperlessBilling', 'Churn']

for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
print("✅ Binary columns encoded")

# ---- STEP 6: Encode gender ----
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
print("✅ Gender encoded")

# ---- STEP 7: One-Hot Encode multi-category columns ----
multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
              'OnlineBackup', 'DeviceProtection', 'TechSupport',
              'StreamingTV', 'StreamingMovies', 'Contract',
              'PaymentMethod']

df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
print("✅ Multi-category columns encoded")

# ---- STEP 8: Scale numerical columns ----
scaler = StandardScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
    df[['tenure', 'MonthlyCharges', 'TotalCharges']]
)
print("✅ Numerical columns scaled")

# ---- STEP 9: Final check ----
print("\n=== FINAL SHAPE ===")
print(df.shape)
print("\n=== FIRST 3 ROWS ===")
print(df.head(3))
print("\n=== ANY MISSING VALUES? ===")
print(df.isnull().sum().sum(), "missing values remaining")

# ---- STEP 10: Save cleaned data ----
df.to_csv('churn_cleaned.csv', index=False)
print("\n✅ Cleaned data saved as churn_cleaned.csv")

# ---- STEP 11: Save the scaler ----
import joblib
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved as scaler.pkl")