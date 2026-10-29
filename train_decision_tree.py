
import pandas as pd
import joblib
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier

# Load training data
train_data = pd.read_csv("data/train_data.csv")

# Separate features and target
X_train = train_data.drop(columns=["Class"])
y_train = train_data["Class"]

# Initialize Decision Tree
model = DecisionTreeClassifier(
    max_depth=12,
    min_samples_split=5,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Create model folder
Path("model").mkdir(exist_ok=True)

# Save trained model
joblib.dump(model, "model/decision_tree.joblib")

print("Decision Tree training completed.")
print("Training samples:", len(X_train))
print("Input features:", X_train.shape[1])
print("Classes:", list(model.classes_))
print("Tree depth:", model.get_depth())
print("Leaf nodes:", model.get_n_leaves())
print("Model saved: model/decision_tree.joblib")
