import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("college_messages.csv")

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

def predict_message(msg):
    msg_vec = vectorizer.transform([msg])
    return model.predict(msg_vec)[0]
