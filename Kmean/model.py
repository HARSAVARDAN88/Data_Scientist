import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

# Load data
df = pd.read_csv("customer_kmeans.csv")

# Features
X = df[["age", "monthly_spend", "visits_per_month"]]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Save model & scaler
pickle.dump(kmeans, open("kmeans_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("K-Means model trained & saved")
