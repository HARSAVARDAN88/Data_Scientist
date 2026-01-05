import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("mobile_health.csv")

X = df[['screen_time_hours', 'sleep_hours', 'eye_strain']]
y = df['health_risk']

# Train model
model = LogisticRegression()
model.fit(X, y)

def predict_health(screen, sleep, eye):
    return model.predict([[screen, sleep, eye]])[0]
