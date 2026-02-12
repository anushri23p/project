# ===============================
# AI-Based Spam Email Detection
# ===============================

import pandas as pd
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

nltk.download('stopwords')

# ===============================
# 1. Dataset (Built-in)
# ===============================
data = {
    "label": [
        "spam","ham","spam","ham","spam","ham","spam","ham","spam","ham"
    ],
    "text": [
        "Win money now",
        "Are we meeting today",
        "Congratulations you won a prize",
        "Please call me later",
        "Claim your free reward",
        "Let's study together",
        "Urgent offer click now",
        "How are you doing",
        "Free entry in competition",
        "See you tomorrow"
    ]
}

df = pd.DataFrame(data)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# ===============================
# 2. Text Preprocessing
# ===============================
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

# ===============================
# 3. Feature Extraction
# ===============================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# ===============================
# 4. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5. Model Training
# ===============================
model = MultinomialNB()
model.fit(X_train, y_train)

# ===============================
# 6. Evaluation
# ===============================
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, zero_division=0))

# ===============================
# 7. Prediction Function
# ===============================
def predict_email(email):
    email = clean_text(email)
    email_vec = vectorizer.transform([email])
    result = model.predict(email_vec)
    return "SPAM" if result[0] == 1 else "NOT SPAM"

# ===============================
# 8. Command Line Interface
# ===============================
print("\n--- Spam Detection System ---")
while True:
    msg = input("Enter email (or exit): ")
    if msg.lower() == "exit":
        break
    print("Prediction:", predict_email(msg))