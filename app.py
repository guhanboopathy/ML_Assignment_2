
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Dry Bean Classification Lab",
    page_icon="🫘",
    layout="wide"
)

st.title("Dry Bean Classification Lab")
st.write(
    "Compare five machine learning classifiers "
    "using a labeled Dry Bean test dataset."
)

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbors": "model/knn.joblib",
    "Gaussian Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

st.subheader("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV containing 16 features and the Class column",
    type=["csv"]
)

st.subheader("2. Select Classification Model")

selected_model = st.selectbox(
    "Choose a trained classifier",
    list(MODEL_PATHS.keys())
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if "Class" not in data.columns:
            st.error("Missing target column: Class")
            st.stop()

        model = load_model(MODEL_PATHS[selected_model])

        required_features = list(model.feature_names_in_)
        missing = [
            column for column in required_features
            if column not in data.columns
        ]
        extra = [
            column for column in data.columns
            if column not in required_features + ["Class"]
        ]

        if missing or extra:
            st.error(
                f"Invalid CSV schema. Missing: {missing}; "
                f"Unexpected: {extra}"
            )
            st.stop()

        if data.empty:
            st.error("The uploaded CSV contains no records.")
            st.stop()

        X_test = data[required_features].copy()
        y_test = data["Class"]

        X_test = X_test.apply(pd.to_numeric, errors="coerce")

        if X_test.isna().any().any() or y_test.isna().any():
            st.error("Missing or non-numeric feature values detected.")
            st.stop()

        unknown_classes = set(y_test) - set(model.classes_)
        if unknown_classes:
            st.error(f"Unknown target classes: {unknown_classes}")
            st.stop()

        st.success(
            f"Loaded {len(data)} records with "
            f"{len(required_features)} features."
        )

        # Use saved model only — no training
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, average="macro",
                zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, average="macro",
                zero_division=0
            ),
            "F1": f1_score(
                y_test, y_pred, average="macro",
                zero_division=0
            ),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }

        # AUC requires all seven classes in the test data
        if set(y_test) == set(model.classes_):
            metrics["AUC"] = roc_auc_score(
                y_test,
                y_prob,
                labels=model.classes_,
                multi_class="ovr",
                average="macro"
            )
        else:
            metrics["AUC"] = None

        st.subheader("3. Evaluation Metrics")

        columns = st.columns(3)

        for index, (name, value) in enumerate(metrics.items()):
            label = f"{value:.4f}" if value is not None else "N/A"
            columns[index % 3].metric(name, label)

        st.subheader("4. Confusion Matrix")

        cm = confusion_matrix(
            y_test, y_pred, labels=model.classes_
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            ax=ax
        )
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("5. Classification Report")

        report = classification_report(
            y_test, y_pred,
            labels=model.classes_,
            output_dict=True,
            zero_division=0
        )

        st.dataframe(pd.DataFrame(report).transpose())

        st.caption(
            "Evaluation uses pre-trained models only. "
            "No retraining is performed on uploaded data."
        )

    except Exception as error:
        st.error(f"Evaluation failed: {error}")
else:
    st.info("Upload test_data.csv to begin evaluation.")
