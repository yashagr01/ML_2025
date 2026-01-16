# Decision Tree for PlayTennis Dataset

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# -------------------------
# Create the dataset
# -------------------------
data = {
    'Outlook': ['Rainy','Rainy','Overcast','Sunny','Sunny','Sunny','Overcast','Rainy',
                'Rainy','Sunny','Rainy','Overcast','Overcast','Sunny'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild',
                    'Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High',
                 'Normal','Normal','Normal','High','Normal','High'],
    'Windy': ['False','True','False','False','False','True','True','False',
              'False','False','True','True','False','True'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes','No',
                    'Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)

# -------------------------
# Encode categorical data
# -------------------------
encoder = LabelEncoder()
for column in df.columns:
    df[column] = encoder.fit_transform(df[column])

# -------------------------
# Split features and target
# -------------------------
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# -------------------------
# Train Decision Tree
# -------------------------
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,        # prevents overlapping
    random_state=42
)

model.fit(X, y)

# -------------------------
# Plot Decision Tree
# -------------------------
plt.figure(figsize=(24, 12))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True,
    fontsize=10
)

plt.title("Decision Tree for Play Tennis Dataset", fontsize=14)
plt.show()
