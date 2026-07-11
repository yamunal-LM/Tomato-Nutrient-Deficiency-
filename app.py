import streamlit as st
import pandas as pd
import joblib

# Load trained files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="Tomato Nutrient Deficiency Prediction",
    page_icon="🍅"
)

st.title("🍅 Tomato Nutrient Deficiency Prediction System")
st.write("Enter the tomato leaf details below and click Predict.")

sample_id = st.number_input("Sample ID", value=201)

leaf_color = st.selectbox(
    "Leaf Color",
    ["Green", "Yellow", "Light Green", "Dark Green"]
)

yellowing = st.selectbox(
    "Yellowing",
    ["Low", "Medium", "High"]
)

brown_spots = st.selectbox(
    "Brown Spots",
    ["Yes", "No"]
)

leaf_curl = st.selectbox(
    "Leaf Curl",
    ["Yes", "No"]
)

vein_color = st.selectbox(
    "Vein Color",
    ["Green", "Yellow", "Light Green"]
)

if st.button("Predict Nutrient Deficiency"):

    new_data = pd.DataFrame({
        "Sample_ID": [sample_id],
        "Leaf_Color": [leaf_color],
        "Yellowing": [yellowing],
        "Brown_Spots": [brown_spots],
        "Leaf_Curl": [leaf_curl],
        "Vein_Color": [vein_color]
    })

    new_data = pd.get_dummies(new_data)
    new_data = new_data.reindex(columns=feature_columns, fill_value=0)
    new_data = scaler.transform(new_data)

    prediction = model.predict(new_data)

    st.success(f"Predicted Nutrient Deficiency: {prediction[0]}")
