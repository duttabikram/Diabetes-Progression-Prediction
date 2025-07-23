import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("Diabetes Progression Prediction")

st.markdown("""
This app predicts the **Diabetes Disease Progression** using a **Decision Tree Regressor** model.
""")

# Input form
feature_names = [
    "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"
]

user_input = []
for feature in feature_names:
    if feature == "sex":
        sex_input = st.selectbox("Sex", ["Male", "Female"])
        # Map to numeric values used in sklearn's diabetes dataset
        val = 0.05068012 if sex_input == "Male" else -0.04464164
    else:
        val = st.number_input(f"{feature}", value=0.0, format="%.5f")
    
    user_input.append(val)

if st.button("Predict Disease Progression"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)
    st.success(f"Predicted Disease Progression Value: {prediction[0]:.2f}")
