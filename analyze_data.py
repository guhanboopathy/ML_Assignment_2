import pandas as pd

# Load dataset
df = pd.read_excel("data/Dry_Bean_Dataset.xlsx")

print("\n===== DATASET INFORMATION =====")
print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== FIRST FIVE ROWS =====")
print(df.head())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE RECORDS =====")
print("Duplicate rows:", df.duplicated().sum())

print("\n===== CLASS DISTRIBUTION =====")
print(df["Class"].value_counts())

print("\n===== FEATURE STATISTICS =====")
print(df.drop(columns=["Class"]).describe())
