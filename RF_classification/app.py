import streamlit as st
import pickle
import numpy as np
import os

# Load model safely
path = os.path.join(os.path.dirname(__file__), "student_model.pkl")
model = pickle.load(open(path, "rb"))

st.title("🎓 Student Pass Prediction")
st.write("Random Forest Classification")

study_hours = st.number_input("Study Hours per Day", 0, 12, 5)
attendance = st.number_input("Attendance (%)", 0, 100, 75)
internal = st.number_input("Internal Marks", 0, 100, 50)

if st.button("Predict"):
    data = np.array([[study_hours, attendance, internal]])
    result = model.predict(data)

    if result[0] == 1:
        st.success("✅ Student will PASS")
    else:
        st.error("❌ Student will FAIL")
