import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load training data
train_data = pd.read_csv("data/train_data.csv")

# Separate features and target
X_train = train_data.drop(columns=["Class"])
y_train = train_data["Class"]

# Create pipeline with scaling and KNN
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="minkowski",
        p=2
    ))
])

# Train model
model.fit(X_train, y_train)

# Create model directory
Path("model").mkdir(exist_ok=True)

# Save trained pipeline
joblib.dump(model, "model/knn.joblib")

print("KNN training completed.")
print("Training samples:", len(X_train))
print("Input features:", X_train.shape[1])
print("Classes:", list(model.classes_))
print("K value:", model.named_steps["classifier"].n_neighbors)
print("Distance metric: Euclidean")
print("Weights: distance")
print("Model saved: model/knn.joblib")
