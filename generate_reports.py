
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay
)

# Load held-out test data
test_data = pd.read_csv("test_data.csv")

X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]

models = {
    "logistic_regression": "model/logistic_regression.joblib",
    "decision_tree": "model/decision_tree.joblib",
    "knn": "model/knn.joblib",
    "naive_bayes": "model/naive_bayes.joblib",
    "svm": "model/svm.joblib"
}

# Create output folders
Path("results/reports").mkdir(parents=True, exist_ok=True)
Path("results/confusion_matrices").mkdir(
    parents=True, exist_ok=True
)

for name, model_path in models.items():

    model = joblib.load(model_path)

    # Predict without retraining
    y_pred = model.predict(X_test)

    # Generate classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=model.classes_,
        zero_division=0
    )

    report_path = f"results/reports/{name}.txt"

    with open(report_path, "w") as file:
        file.write(report)

    # Generate confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=model.classes_,
        display_labels=model.classes_,
        cmap="Blues",
        xticks_rotation=45,
        ax=ax
    )

    ax.set_title(f"{name} - Confusion Matrix")
    fig.tight_layout()

    image_path = (
        f"results/confusion_matrices/{name}.png"
    )

    fig.savefig(image_path, dpi=150)
    plt.close(fig)

    print(f"{name}: report and confusion matrix saved")

print("\nAll five models processed successfully.")