import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# App Title
st.set_page_config(page_title="Electricity Bill Prediction", layout="centered")
st.title("⚡ Electricity Bill Prediction App")

st.write("This app predicts monthly electricity bill using Linear Regression.")

# Load Dataset
df = pd.read_csv("electricity_bill_dataset.csv")

# Show dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df)

# Features & Target
X = df[['Units', 'Appliances', 'Hours_per_day']]
y = df['Bill']

# Train model
model = LinearRegression()
model.fit(X, y)

st.subheader("🔢 Enter Details")

# User Inputs
units = st.number_input("Electricity Units Consumed", min_value=0, value=200)
appliances = st.number_input("Number of Appliances", min_value=0, value=5)
hours = st.number_input("Hours of Usage per Day", min_value=0, value=6)

# Prediction
if st.button("Predict Bill"):
    input_data = [[units, appliances, hours]]
    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Electricity Bill: ₹ {prediction[0]:.2f}")

# Footer
st.markdown("---")
st.markdown("📘 **Linear Regression Mini Project**")
