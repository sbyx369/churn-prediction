# ================================
# CUSTOMER CHURN PREDICTION SYSTEM
# Phase 4 - Model Training
# ================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---- STEP 1: Load cleaned data ----
df = pd.read_csv('churn_cleaned.csv')
scaler = joblib.load('scaler.pkl') if __import__('os').path.exists('scaler.pkl') else None
print("✅ Cleaned data loaded:", df.shape)

# ---- STEP 2: Split features and target ----
X = df.drop('Churn', axis=1)  # Everything except Churn
y = df['Churn']                # Only Churn column
print("✅ Features:", X.shape)
print("✅ Target:", y.shape)

from imblearn.over_sampling import SMOTE

# ---- STEP 3: Split into train and test sets ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n✅ Training set:", X_train.shape)
print("✅ Testing set:", X_test.shape)

# ---- STEP 3B: Apply SMOTE to training data only ----
print("\n⏳ Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("✅ SMOTE applied!")
print("New training distribution:")
print(pd.Series(y_train).value_counts())

# ---- STEP 4: Train Logistic Regression ----
print("\n⏳ Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
print("✅ Logistic Regression trained!")

# ---- STEP 5: Train Random Forest ----
print("\n⏳ Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("✅ Random Forest trained!")

# ---- STEP 6: Evaluate both models ----
def evaluate(name, y_test, preds):
    print(f"\n=== {name} ===")
    print(f"Accuracy  : {accuracy_score(y_test, preds):.4f}")
    print(f"Precision : {precision_score(y_test, preds):.4f}")
    print(f"Recall    : {recall_score(y_test, preds):.4f}")
    print(f"F1 Score  : {f1_score(y_test, preds):.4f}")

evaluate("Logistic Regression", y_test, lr_preds)
evaluate("Random Forest",       y_test, rf_preds)

# ---- STEP 7: Confusion Matrix for both ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, preds, title in zip(axes,
                             [lr_preds, rf_preds],
                             ['Logistic Regression', 'Random Forest']):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{title} - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.show()

# ---- STEP 8: Feature Importance (Random Forest) ----
feat_importance = pd.Series(rf.feature_importances_,
                            index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_importance.head(10).plot(kind='bar', color='steelblue')
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.show()

# ---- STEP 9: Save the model ----
import joblib

joblib.dump(lr, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model saved as churn_model.pkl")