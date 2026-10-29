import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load original dataset
df = pd.read_excel("data/Dry_Bean_Dataset.xlsx")

print("Original dataset shape:", df.shape)

# Remove exact duplicates
df = df.drop_duplicates().reset_index(drop=True)

print("After removing duplicates:", df.shape)

# Separate input features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# Split data with class proportions preserved
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create data directory
Path("data").mkdir(exist_ok=True)

# Save training data
train_data = X_train.copy()
train_data["Class"] = y_train
train_data.to_csv("data/train_data.csv", index=False)

# Save test data WITH target column
test_data = X_test.copy()
test_data["Class"] = y_test
test_data.to_csv("test_data.csv", index=False)

print("\n===== SPLIT RESULTS =====")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of features:", X_train.shape[1])

print("\n===== TRAINING CLASS DISTRIBUTION =====")
print(y_train.value_counts())

print("\n===== TEST CLASS DISTRIBUTION =====")
print(y_test.value_counts())

print("\nFiles created:")
print("data/train_data.csv")
print("test_data.csv")
