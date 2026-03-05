# ================================
# CUSTOMER CHURN PREDICTION SYSTEM
# Phase 2 - Exploratory Data Analysis
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- STEP 1: Load the dataset ----
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# ---- STEP 2: First look at the data ----
print("=== SHAPE (rows, columns) ===")
print(df.shape)

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== CHURN DISTRIBUTION ===")
print(df['Churn'].value_counts())

# ---- STEP 3: Visualize Churn Distribution ----
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()

# ---- STEP 4: Churn by Contract Type ----
plt.figure(figsize=(8, 4))
sns.countplot(x='Contract', hue='Churn', data=df, palette='Set2')
plt.title('Churn by Contract Type')
plt.tight_layout()
plt.show()

# ---- STEP 5: Monthly Charges Distribution ----
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', bins=30, palette='Set2')
plt.title('Monthly Charges vs Churn')
plt.tight_layout()
plt.show()