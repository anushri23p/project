import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("C:\\Users\\ARATI\\Desktop\\internship\\diabetes.csv")
print(data.head())
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in cols_with_zero:
    data[col] = data[col].replace(0, data[col].mean())

X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\n************* Disease Prediction System *************")


user_data = []

features = X.columns.tolist()

for feature in features:
    value = float(input(f"Enter {feature}: "))
    user_data.append(value)

user_data = np.array(user_data).reshape(1, -1)
user_data_scaled = scaler.transform(user_data)

prediction = model.predict(user_data_scaled)

if prediction[0] == 1:
    print("\nPrediction: HIGH RISK of Disease")
else:
    print("\nPrediction: LOW RISK of Disease")

print("""
DISCLAIMER:
This AI-based disease prediction system is developed for educational purposes only.
It does NOT replace professional medical advice, diagnosis, or treatment.
User data is not stored or shared.
""")