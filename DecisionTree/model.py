import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("bike_loan_decision_tree.csv")

X = df[['income', 'age', 'credit_score']]
y = df['loan_approved']

model = DecisionTreeClassifier()
model.fit(X, y)

def predict_loan(income, age, credit):
    return model.predict([[income, age, credit]])[0]
