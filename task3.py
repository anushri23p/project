import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('stopwords')

data = pd.read_csv("C:\\Users\\ARATI\\Desktop\\intership\\reviews.csv")


print("Dataset Preview:")
print(data.head())
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def rating_to_sentiment(rating):
    if rating >= 7:
        return "positive"
    elif rating <= 4:
        return "negative"
    else:
        return "neutral"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

data['sentiment'] = data['Ratings'].apply(rating_to_sentiment)

data['cleaned_review'] = data['Comments'].apply(preprocess_text)


vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data['cleaned_review'])
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

data['sentiment'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

def predict_sentiment(text):
    text = preprocess_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return prediction[0]

while True:
    user_input = input("\nEnter a review (or type 'exit'): ")
    if user_input.lower() == 'exit':
        break
    print("Predicted Sentiment:", predict_sentiment(user_input))