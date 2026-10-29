
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load training data
train_data = pd.read_csv("data/train_data.csv")

# Separate features and target
X_train = train_data.drop(columns=["Class"])
y_train = train_data["Class"]

# Create a pipeline
# Scaling is fitted only on training data
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

# Train the model
model.fit(X_train, y_train)

# Create model folder
Path("model").mkdir(exist_ok=True)

# Save complete trained pipeline
joblib.dump(model, "model/logistic_regression.joblib")

print("Logistic Regression training completed.")
print("Training samples:", len(X_train))
print("Input features:", X_train.shape[1])
print("Classes:", list(model.classes_))
print("Model saved: model/logistic_regression.joblib")
