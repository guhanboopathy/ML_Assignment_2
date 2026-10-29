import pandas as pd
import joblib
from pathlib import Path

from sklearn.naive_bayes import GaussianNB

# Load training data
train_data = pd.read_csv("data/train_data.csv")

X_train = train_data.drop(columns=["Class"])
y_train = train_data["Class"]

# Initialize Gaussian Naive Bayes
model = GaussianNB()

# Train model
model.fit(X_train, y_train)

# Create model directory
Path("model").mkdir(exist_ok=True)

# Save trained model
joblib.dump(model, "model/naive_bayes.joblib")

print("Gaussian Naive Bayes training completed.")
print("Training samples:", len(X_train))
print("Input features:", X_train.shape[1])
print("Classes:", list(model.classes_))
print("Model saved: model/naive_bayes.joblib")
