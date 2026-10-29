from pathlib import Path
import io

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbors": "model/knn.joblib",
    "Gaussian Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}

st.set_page_config(page_title="Dry Bean Classification Lab", page_icon="🫘", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def load_model(path):
    return joblib.load(BASE_DIR / path)


@st.cache_data(show_spinner=False)
def evaluate_upload(csv_bytes):
    data = pd.read_csv(io.BytesIO(csv_bytes))
    reference = load_model(MODEL_PATHS["Logistic Regression"])
    features = list(reference.feature_names_in_)
    if "Class" not in data.columns:
        raise ValueError("Missing target column: Class")
    missing = [column for column in features if column not in data.columns]
    extra = [column for column in data.columns if column not in features + ["Class"]]
    if missing or extra:
        raise ValueError(f"Invalid CSV schema. Missing: {missing}; Unexpected: {extra}")
    if data.empty:
        raise ValueError("The uploaded CSV contains no records.")
    X_test = data[features].apply(pd.to_numeric, errors="coerce")
    y_test = data["Class"]
    if X_test.isna().any().any() or y_test.isna().any():
        raise ValueError("Missing or non-numeric feature values detected.")
    unknown = set(y_test) - set(reference.classes_)
    if unknown:
        raise ValueError(f"Unknown target classes: {sorted(unknown)}")

    rows, details = [], {}
    for name, path in MODEL_PATHS.items():
        model = load_model(path)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        auc = None
        if set(y_test) == set(model.classes_):
            auc = roc_auc_score(y_test, probabilities, labels=model.classes_,
                                multi_class="ovr", average="macro")
        metrics = {
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": auc,
            "Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
            "Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
            "F1": f1_score(y_test, predictions, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_test, predictions),
        }
        rows.append({"ML Model Name": name, **metrics})
        details[name] = {
            "predictions": pd.DataFrame({
                "Data row": range(1, len(data) + 1),
                "Actual Class": y_test.to_numpy(),
                "Predicted Class": predictions,
                "Correct": y_test.to_numpy() == predictions,
            }),
            "metrics": metrics,
            "classes": model.classes_,
            "matrix": confusion_matrix(y_test, predictions, labels=model.classes_),
            "report": pd.DataFrame(classification_report(
                y_test, predictions, labels=model.classes_, output_dict=True,
                zero_division=0)).transpose(),
        }
    return pd.DataFrame(rows), details, len(data), len(features)


st.caption("ML Assignment 2 · Guhan Boopathy · 2026AK05428")
st.title("Dry Bean Classification Lab")
st.write("Explore seven bean varieties. Compare five trained classifiers on the same labeled test data.")
st.caption("5 models · 6 evaluation metrics · 7 bean classes · No retraining")
st.divider()

st.subheader("Dataset overview")
st.markdown("[Dry Bean Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)")
columns = st.columns(4)
for column, label, value in zip(columns, ["Input features", "Bean classes", "Training samples", "Test samples"],
                               [16, 7, 10834, 2709]):
    column.metric(label, f"{value:,}", border=True)
st.caption("13,611 original records; 68 duplicates removed. The 13,543 cleaned records use an 80:20 stratified training/test split (random state 42).")

with st.sidebar:
    st.header("ML Assignment 2")
    st.caption("Guhan Boopathy · 2026AK05428")
    st.divider()
    st.subheader("Test data")
    st.write("Download the sample, then upload a labeled CSV to evaluate all five saved models.")
    st.download_button("Download test_data.csv", data=(BASE_DIR / "test_data.csv").read_bytes(),
                       file_name="test_data.csv", mime="text/csv")
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
    st.caption("CSV only · Include the Class target column")
    with st.expander("Required CSV format"):
        reference = load_model(MODEL_PATHS["Logistic Regression"])
        st.write("Include all 16 numeric feature columns and the Class target column. Column order can differ; extra columns are not accepted.")
        st.code(",".join(list(reference.feature_names_in_) + ["Class"]), language="text")
        st.write("Valid Class values: " + ", ".join(reference.classes_))
        st.caption("AUC is available only when all seven classes are present.")
    st.divider()
    st.caption("The sample contains held-out test records. All models evaluate the same uploaded rows without retraining.")

if uploaded_file is None:
    st.info("Use the Test data sidebar to download and upload a sample. On a small screen, open the sidebar using the top-left arrow.")
else:
    try:
        with st.spinner("Evaluating all five saved models…"):
            comparison, details, record_count, feature_count = evaluate_upload(uploaded_file.getvalue())
    except Exception as error:
        st.error(f"Evaluation failed: {error}")
        st.stop()

    st.success(f"Loaded {record_count} records with {feature_count} features.")
    st.subheader("Explore results")
    selected = st.selectbox("Choose a trained classifier", list(MODEL_PATHS))
    result = details[selected]
    st.subheader("Model comparison")
    st.caption("All models use the same uploaded rows. Teal cells mark the highest score in each column, including ties. Precision, Recall, F1 and one-vs-rest AUC use macro averaging.")
    metric_names = list(comparison.columns[1:])
    styled_comparison = comparison.style.format(
        {name: "{:.4f}" for name in metric_names}, na_rep="N/A"
    ).highlight_max(subset=metric_names, axis=0,
                    props="background-color: #d4f3e9; color: #123d35; font-weight: bold;")
    st.dataframe(styled_comparison, hide_index=True, width="stretch")
    st.subheader(f"{selected} metrics")
    columns = st.columns(3)
    for index, (name, value) in enumerate(result["metrics"].items()):
        columns[index % 3].metric(name, f"{value:.4f}" if value is not None else "N/A", border=True)

    st.divider()
    st.subheader(f"{selected} · Confusion matrix")
    st.caption("Rows show actual classes; columns show predicted classes. The diagonal contains correct predictions.")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(result["matrix"], annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=result["classes"], yticklabels=result["classes"], ax=ax)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader(f"{selected} · Classification report")
    st.dataframe(result["report"].style.format("{:.4f}"), width="stretch")
    st.caption("Support is the number of actual records in each class.")

    st.divider()
    st.subheader(f"{selected} · Predictions and errors")
    predictions = result["predictions"]
    error_count = int((~predictions["Correct"]).sum())
    correct_column, error_column = st.columns(2)
    correct_column.metric("Correct predictions", f"{record_count - error_count:,}", border=True)
    error_column.metric("Misclassified rows", f"{error_count:,}", border=True)
    st.caption("Data row starts at 1 after the CSV header. Incorrect predictions have a pink background and an Incorrect status.")
    errors_only = st.checkbox("Show misclassified rows only")
    visible_predictions = predictions.loc[~predictions["Correct"]] if errors_only else predictions
    if visible_predictions.empty:
        st.info("No misclassified rows for this model on the uploaded data.")
    else:
        display_predictions = visible_predictions.assign(
            Status=visible_predictions["Correct"].map({True: "Correct", False: "Incorrect"})
        ).drop(columns="Correct")
        styled_predictions = display_predictions.style.apply(
            lambda row: ["background-color: #fde8eb; color: #742331;" if row["Status"] == "Incorrect" else ""] * len(row),
            axis=1,
        )
        st.dataframe(styled_predictions, hide_index=True, width="stretch")

    st.divider()
    st.caption("ML Assignment 2 · Guhan Boopathy · 2026AK05428 | Evaluation uses saved models only. No retraining.")
