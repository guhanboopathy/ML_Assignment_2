
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# Load held-out test data
test_data = pd.read_csv("test_data.csv")

X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]

# All five saved models
models = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}

results = []

print("Test samples:", len(X_test))
print("Input features:", X_test.shape[1])

for name, path in models.items():

    # Load trained model without retraining
    model = joblib.load(path)

    # Predict class labels and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Calculate all six mandatory metrics
    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(
            y_test,
            y_proba,
            labels=model.classes_,
            multi_class="ovr",
            average="macro"
        ),
        "Precision": precision_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        ),
        "F1": f1_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

# Create results table
results_df = pd.DataFrame(results)

# Save exact results for documentation and app
Path("results").mkdir(exist_ok=True)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

print("\nResults saved: results/model_comparison.csv")
print("All models evaluated without retraining.")
