import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------------------------------------------
# 1. Load dataset
# ----------------------------------------------------
data = pd.DataFrame({
    'Outlook': ['Rainy','Rainy','Overcast','Sunny','Sunny','Sunny','Overcast','Rainy','Rainy','Sunny','Rainy','Overcast','Overcast','Sunny'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
    'Windy': ['False','True','False','False','False','True','True','False','False','False','True','True','False','True'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
})

# ----------------------------------------------------
# 2. Encode categorical variables
# ----------------------------------------------------
encoders = {}
for col in data.columns:
    encoders[col] = LabelEncoder()
    data[col] = encoders[col].fit_transform(data[col])

X = data.drop('PlayTennis', axis=1)
y = data['PlayTennis']

# ----------------------------------------------------
# 3. Manual Naive Bayes
# ----------------------------------------------------
def naive_bayes_predict_manual(row):
    classes = np.unique(y)
    probabilities = {}

    for c in classes:
        prior = len(y[y == c]) / len(y)
        likelihood = 1

        for col, value in zip(X.columns, row):
            num = len(data[(data[col] == value) & (data['PlayTennis'] == c)])
            den = len(data[data['PlayTennis'] == c])
            likelihood *= num / den if den != 0 else 0

        probabilities[c] = prior * likelihood

    return max(probabilities, key=probabilities.get)

# ----------------------------------------------------
# 4. Train Sklearn Naive Bayes
# ----------------------------------------------------
model = MultinomialNB()
model.fit(X, y)

# ----------------------------------------------------
# 5. Take user input for new instance
# ----------------------------------------------------
print("\nEnter values for prediction:")

outlook = input("Outlook (Sunny / Rainy / Overcast): ")
temperature = input("Temperature (Hot / Mild / Cool): ")
humidity = input("Humidity (High / Normal): ")
windy = input("Windy (True / False): ")

user_input = [outlook, temperature, humidity, windy]

# Encode user input
encoded_input = [
    encoders[col].transform([val])[0]
    for col, val in zip(X.columns, user_input)
]

# ----------------------------------------------------
# 6. Predictions
# ----------------------------------------------------
manual_pred = naive_bayes_predict_manual(encoded_input)
sklearn_pred = model.predict([encoded_input])[0]

# Decode result (0/1 → No/Yes)
result_manual = encoders['PlayTennis'].inverse_transform([manual_pred])[0]
result_sklearn = encoders['PlayTennis'].inverse_transform([sklearn_pred])[0]

# ----------------------------------------------------
# 7. Output
# ----------------------------------------------------
print("\nPrediction Results:")
print("Manual Naive Bayes Prediction:", result_manual)
print("Sklearn Naive Bayes Prediction:", result_sklearn)