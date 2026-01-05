import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("ebike_model.pkl", "rb"))

st.title("⚡ E-Bike Range Prediction")
st.write("Random Forest Regression Model")

battery = st.number_input("Battery Capacity (Wh)", 300, 700, 500)
speed = st.number_input("Average Speed (km/h)", 15, 40, 28)
weight = st.number_input("Rider Weight (kg)", 50, 120, 75)
terrain = st.selectbox("Terrain Type", ["Flat", "Hilly"])

terrain = 0 if terrain == "Flat" else 1

if st.button("Predict Range"):
    data = np.array([[battery, speed, weight, terrain]])
    prediction = model.predict(data)
    st.success(f"🔋 Estimated Range: {prediction[0]:.2f} km")
