# -----------------------------------------
# Decision Tree for Lawn Tractor Ownership
# -----------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -----------------------------------------
# 1. Create Dataset
# -----------------------------------------

data = {
    "Income": [60, 75, 85.5, 52.8, 64.8, 64.8, 61.5, 43.2, 87, 84,
               110.1, 49.2, 108, 59.2, 82.8, 66, 69, 47.4, 93, 33,
               51, 51, 81, 63],
    "Age": [18.4, 19.6, 16.8, 20.8, 21.6, 17.2, 20.8, 20.4, 23.6, 17.6,
            19.2, 17.6, 17.6, 16, 22.4, 18.4, 20, 16.4, 20.8, 18.8,
            22, 14, 20, 14.8],
    "Owner": ["Owner", "Nonowner", "Owner", "Nonowner", "Owner", "Nonowner",
              "Owner", "Nonowner", "Owner", "Nonowner", "Owner", "Nonowner",
              "Owner", "Nonowner", "Owner", "Nonowner", "Owner", "Nonowner",
              "Owner", "Nonowner", "Owner", "Nonowner", "Owner", "Nonowner"]
}

df = pd.DataFrame(data)

# -----------------------------------------
# 2. Encode Target Variable
# -----------------------------------------

encoder = LabelEncoder()
df["Owner"] = encoder.fit_transform(df["Owner"])  # Owner=1, Nonowner=0

X = df[["Income", "Age"]]
y = df["Owner"]

# -----------------------------------------
# 3. Train-Test Split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------------------
# 4. Train Pruned Decision Tree
# -----------------------------------------

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------------
# 5. Model Evaluation
# -----------------------------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# -----------------------------------------
# 6. Visualize Decision Tree (No Overlap)
# -----------------------------------------

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=["Income", "Age"],
    class_names=["Nonowner", "Owner"],
    filled=True,
    fontsize=12
)
plt.show()

# -----------------------------------------
# 7. Print Decision Rules
# -----------------------------------------

rules = export_text(model, feature_names=["Income", "Age"])
print("\nDecision Tree Rules:\n")
print(rules)
