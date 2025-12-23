import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)

# -------------------- TITLE --------------------
st.title("🌾 Crop Recommendation System")
st.write("Predict the best crop using KNN Machine Learning Algorithm")

# -------------------- LOAD DATA --------------------
# IMPORTANT: crop_data.csv MUST be in the same folder as this app.py
df = pd.read_csv("crop_data.csv")

# -------------------- DATA PREVIEW --------------------
with st.expander("📊 View Dataset"):
    st.dataframe(df)

# -------------------- FEATURES & TARGET --------------------
X = df.drop("Crop", axis=1)
y = df["Crop"]

# -------------------- TRAIN TEST SPLIT --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -------------------- FEATURE SCALING --------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------- KNN MODEL --------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# -------------------- MODEL ACCURACY --------------------
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

# -------------------- SIDEBAR INPUTS --------------------
st.sidebar.header("🌱 Enter Soil & Climate Values")

N = st.sidebar.number_input("Nitrogen (N)", 0, 150, 50)
P = st.sidebar.number_input("Phosphorus (P)", 0, 150, 45)
K = st.sidebar.number_input("Potassium (K)", 0, 150, 40)
temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 60.0)

# -------------------- PREDICTION --------------------
if st.sidebar.button("🌾 Predict Crop"):
    input_data = [[N, P, K, temperature, humidity]]
    input_scaled = scaler.transform(input_data)
    prediction = knn.predict(input_scaled)

    st.balloons()
    st.success(f"✅ Recommended Crop: **{prediction[0]}**")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("👨‍💻 Built using **KNN Algorithm & Streamlit**")
