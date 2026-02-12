import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("C:\\Users\\ARATI\\Desktop\\internship\\transactions.csv")

# Basic inspection
print(df.head())
print(df.info())

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Features & label
X = df.drop("Class", axis=1)
y = df["Class"]
from sklearn.ensemble import IsolationForest

anomaly_model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

df["anomaly_score"] = anomaly_model.fit_predict(X)

# -1 = anomaly, 1 = normal
df["anomaly"] = df["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

print(df["anomaly"].value_counts())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

clf.fit(X_train, y_train)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

importances = clf.feature_importances_
features = X.columns

plt.figure(figsize=(10,5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()

# save as app.py
import streamlit as st
import pandas as pd
import joblib

st.title("AI Fraud Detection Dashboard")

model = joblib.load("fraud_model.pkl")

uploaded_file = st.file_uploader("Upload Transaction CSV")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    predictions = model.predict(data)
    data["Fraud Prediction"] = predictions
    st.dataframe(data)