import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Page config
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

# Title
st.title("🌾 Crop Recommendation System")
st.write("Predict the best crop using KNN Machine Learning Algorithm")

# Load dataset
df = pd.read_csv("crop_data.csv")

# Features and target
X = df.drop("Crop", axis=1)
y = df["Crop"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Sidebar inputs
st.sidebar.header("Enter Soil & Climate Values")

N = st.sidebar.number_input("Nitrogen (N)", 0, 150, 50)
P = st.sidebar.number_input("Phosphorus (P)", 0, 150, 45)
K = st.sidebar.number_input("Potassium (K)", 0, 150, 40)
temp = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 60.0)

# Predict button
if st.button("🌱 Predict Crop"):
    input_data = [[N, P, K, temp, humidity]]
    input_scaled = scaler.transform(input_data)
    prediction = knn.predict(input_scaled)

    st.success(f"✅ Recommended Crop: **{prediction[0]}**")

# Show dataset
with st.expander("📊 View Dataset"):
    st.dataframe(df)

