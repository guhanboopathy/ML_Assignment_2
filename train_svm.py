
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# Load training data
train_data = pd.read_csv("data/train_data.csv")

X_train = train_data.drop(columns=["Class"])
y_train = train_data["Class"]

# Create SVM with probability calibration
svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", CalibratedClassifierCV(
        estimator=svm,
        method="sigmoid",
        cv=5,
        ensemble=False
    ))
])

# Train using training data only
model.fit(X_train, y_train)

# Save complete pipeline
Path("model").mkdir(exist_ok=True)
joblib.dump(model, "model/svm.joblib")

print("SVM training completed successfully.")
print("Training samples:", len(X_train))
print("Input features:", X_train.shape[1])
print("Classes:", list(model.classes_))
print("Probability calibration: sigmoid")
print("Model saved: model/svm.joblib")

# Verify probability predictions
probabilities = model.predict_proba(X_train.iloc[:5])

print("Probability matrix shape:", probabilities.shape)
print("Probability sums:", probabilities.sum(axis=1))
