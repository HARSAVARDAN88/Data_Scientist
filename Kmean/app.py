import streamlit as st
import pickle
import numpy as np
import os

# Load files safely
base = os.path.dirname(__file__)
kmeans = pickle.load(open(os.path.join(base, "kmeans_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(base, "scaler.pkl"), "rb"))

st.title("🧩 Customer Clustering (K-Means)")
st.write("Unsupervised Machine Learning")

age = st.number_input("Age", 18, 80, 30)
spend = st.number_input("Monthly Spend (₹)", 1000, 50000, 8000)
visits = st.number_input("Visits per Month", 1, 30, 8)

if st.button("Find Cluster"):
    data = np.array([[age, spend, visits]])
    data_scaled = scaler.transform(data)
    cluster = kmeans.predict(data_scaled)

    st.success(f"Customer belongs to Cluster: {cluster[0]}")
