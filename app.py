import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# =========================
# Load and train the model
# =========================
@st.cache_resource
def load_model():
    data = pd.read_csv("Airbnb.csv")  # Replace with your dataset file

    # Assuming 'price' is the target column
    X = data.drop(columns=['price'])
    y = data['price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    # Save model and scaler
    joblib.dump(rf, "random_forest_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    return rf, scaler, X.columns

model, scaler, feature_names = load_model()

# =========================
# Streamlit UI
# =========================
st.title("🏠 Airbnb Price Prediction")
st.write("Enter property details to predict the price")

inputs = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0)
    inputs.append(value)

if st.button("Predict Price"):
    features = np.array(inputs).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    st.success(f"Predicted Price: ${prediction:.2f}")




