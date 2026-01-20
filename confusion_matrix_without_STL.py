from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Manual confusion matrix
num_classes = len(np.unique(y))
cm_manual = np.zeros((num_classes, num_classes), dtype=int)

for true, pred in zip(y_test, y_pred):
    cm_manual[true][pred] += 1

print("Manual Confusion Matrix:\n", cm_manual)