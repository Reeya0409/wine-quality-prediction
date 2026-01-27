import pandas as pd

import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Wine Quality Prediction", layout="centered")

st.title("🍷 Wine Quality Prediction App")
st.write("Predict whether wine quality is **Good** or **Bad**")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Lab 22 winequality-red.csv")

data = load_data()

# -----------------------------
# Preprocessing
# -----------------------------
# Convert quality to binary classification
# Good wine: quality >= 7
data["quality_label"] = data["quality"].apply(lambda x: 1 if x >= 7 else 0)

X = data.drop(["quality", "quality_label"], axis=1)
y = data["quality_label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter Wine Chemical Properties")

def user_input():
    input_data = {}
    for col in X.columns:
        min_val = float(data[col].min())
        max_val = float(data[col].max())
        mean_val = float(data[col].mean())
        input_data[col] = st.sidebar.slider(
            col, min_val, max_val, mean_val
        )
    return pd.DataFrame([input_data])

input_df = user_input()
input_scaled = scaler.transform(input_df)

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)[0]

st.subheader("🔮 Prediction Result")

if prediction == 1:
    st.success("🍷 This is a **GOOD quality wine**")
else:
    st.error("🍷 This is a **BAD quality wine**")

st.write(f"Confidence: **{max(prediction_proba)*100:.2f}%**")

# -----------------------------
# Model Accuracy
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Accuracy")
st.write(f"Accuracy: **{accuracy*100:.2f}%**")