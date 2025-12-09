import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.DataFrame({
    'Feature_1': [60,62,67,70,71,72,75,78],
    'Feature_2': [22,25,24,20,15,14,14,11],
    'Target': [140,155,159,179,192,200,212,215]
})
print(data.head())
X = data[['Feature_1', 'Feature_2']]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
model = LinearRegression()
model.fit(X_train, y_train)
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
