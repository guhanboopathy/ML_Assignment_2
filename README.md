
# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to implement and compare five machine learning classification models for identifying different varieties of dry beans based on their physical characteristics.

The dataset contains 16 numerical features describing the size and shape of dry beans, including area, perimeter, axis lengths, compactness, and shape factors.

The classification task involves predicting one of seven bean varieties:

- SEKER
- BARBUNYA
- BOMBAY
- CALI
- HOROZ
- SIRA
- DERMASON

### Classification Models Implemented

The following five machine learning algorithms are used:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Model Evaluation

Each classification model is evaluated using six performance metrics:

1. Accuracy
2. Area Under the ROC Curve (AUC)
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)

The objective is to compare the performance of these five algorithms using the same training and testing datasets and identify the best-performing model.

An interactive Streamlit web application is also developed to allow users to upload test data, select a trained classification model, and view evaluation metrics, confusion matrices, and classification reports.

The application loads previously trained models and evaluates uploaded test data without retraining.

---

## 2. Dataset Description

### 2.1 Dataset Information

**Dataset Name:** Dry Bean Dataset

**Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

**Machine Learning Task:** Multiclass Classification

The Dry Bean Dataset contains numerical measurements extracted from images of different dry bean varieties.

These measurements describe the geometric and shape characteristics of beans and are used as input features for classification.

### 2.2 Dataset Characteristics

| Parameter | Value |
|---|---|
| Original Dataset Instances | 13,611 |
| Number of Input Features | 16 |
| Target Variable | Class |
| Number of Target Classes | 7 |
| Missing Values | 0 |
| Duplicate Records | 68 |
| Instances After Cleaning | 13,543 |
| Training Samples | 10,834 |
| Testing Samples | 2,709 |
| Training Percentage | 80% |
| Testing Percentage | 20% |
| Random State | 42 |
| Splitting Method | Stratified Train-Test Split |

### 2.3 Input Features

The dataset contains 16 numerical input features:

1. Area
2. Perimeter
3. MajorAxisLength
4. MinorAxisLength
5. AspectRation
6. Eccentricity
7. ConvexArea
8. EquivDiameter
9. Extent
10. Solidity
11. roundness
12. Compactness
13. ShapeFactor1
14. ShapeFactor2
15. ShapeFactor3
16. ShapeFactor4

These features represent various size, shape, and geometric properties of dry beans.

### 2.4 Target Variable

The target variable is `Class`, which represents the bean variety.

The seven target classes are:

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

### 2.5 Data Preprocessing

The following preprocessing steps were performed:

**Step 1: Data Loading**

The original dataset was loaded using the pandas library from the Excel file.

**Step 2: Data Quality Analysis**

The dataset was examined for:

- Missing values
- Duplicate records
- Feature data types
- Class distribution
- Statistical properties of numerical features

**Step 3: Duplicate Removal**

The original dataset contained 13,611 records.

A total of 68 duplicate records were identified and removed.

The cleaned dataset contains 13,543 records.

**Step 4: Feature and Target Separation**

The dataset was divided into:

- Input features (X): 16 numerical columns
- Target variable (y): Class

**Step 5: Training and Testing Split**

The cleaned dataset was divided using an 80:20 stratified train-test split with `random_state=42`.

- Training samples: 10,834
- Testing samples: 2,709

Stratified sampling was used to maintain approximately the same class distribution in both datasets.

**Step 6: Feature Scaling**

Feature standardization using `StandardScaler` was applied to:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine

Feature scaling was not applied to:

- Decision Tree
- Gaussian Naive Bayes

Scaling was fitted using training data only to prevent data leakage.

**Step 7: Test Data Preparation**

The testing dataset was saved as `test_data.csv`.

The file contains all 16 numerical input features and the `Class` target column.

This file is used for evaluating trained models and testing the Streamlit application without retraining.

## 3. GitHub Repository Link

The complete project source code, trained classification models,
dataset files, evaluation results, and Streamlit application
are maintained in the GitHub repository.

**Repository URL:** [ML_Assignment_2](https://github.com/guhanboopathy/ML_Assignment_2)

**Live application:** [Dry Bean Classification Lab](https://2026ak05428-guhan-boopathy-ml-assignment-2.streamlit.app/)

**Author:** Guhan Boopathy · **BITS ID:** 2026AK05428

### Application features

- Download the held-out test CSV and view the required columns and valid classes.
- Compare all five saved models on the same uploaded records using six metrics, with the highest scores highlighted.
- Inspect the selected model’s metrics, confusion matrix, and classification report.
- Explore actual versus predicted classes and filter misclassified rows.
- View dataset details and use the test-data sidebar in a consistent light theme.

Evaluation uses saved models without retraining. Results appear in vertically stacked sections.

## 4. Classification Models Implemented

Five classification models were implemented using scikit-learn.
All models were trained on the same training dataset containing
10,834 samples and evaluated on the same 2,709 test samples.

### 4.1 Logistic Regression

- Algorithm: Logistic Regression
- Feature scaling: StandardScaler
- Maximum iterations: 2000
- Random state: 42
- Model file: model/logistic_regression.joblib

Logistic Regression learns relationships between input features
and target classes to predict class probabilities.

### 4.2 Decision Tree Classifier

- Algorithm: DecisionTreeClassifier
- Maximum depth: 12
- Minimum samples split: 5
- Random state: 42
- Actual tree depth: 12
- Leaf nodes: 395
- Model file: model/decision_tree.joblib

The Decision Tree uses feature-based splitting rules to
classify bean varieties.

### 4.3 K-Nearest Neighbors (KNN)

- Algorithm: KNeighborsClassifier
- Number of neighbors: 5
- Distance metric: Euclidean
- Weights: Distance
- Feature scaling: StandardScaler
- Model file: model/knn.joblib

KNN predicts the class of a new sample using nearby
training samples.

### 4.4 Gaussian Naive Bayes

- Algorithm: GaussianNB
- Feature scaling: Not applied
- Model file: model/naive_bayes.joblib

Gaussian Naive Bayes uses Bayes' theorem and assumes
that features are conditionally independent given
the target class. It models each feature using a
Gaussian distribution.

### 4.5 Support Vector Machine (SVM)

- Algorithm: Support Vector Classifier
- Kernel: RBF
- Regularization parameter (C): 1.0
- Gamma: scale
- Feature scaling: StandardScaler
- Probability calibration: Sigmoid
- Calibration folds: 5
- Calibration ensemble: False
- Random state: 42
- Model file: model/svm.joblib

SVM identifies decision boundaries between classes.
The RBF kernel supports nonlinear classification.

CalibratedClassifierCV was used to provide class
probability estimates required for multiclass AUC
evaluation.

### Model Storage

All five trained models are saved as .joblib files.

The Streamlit application loads these saved models
to evaluate uploaded test data without retraining.

### Modeling Choices and Rationale

- **Duplicate removal:** The dataset contained 68 exact duplicate records. These were removed before the train-test split to reduce the chance of identical records appearing in both partitions and making test performance look better than it is.
- **Feature scaling:** Dry Bean features have different numerical scales. StandardScaler was used for Logistic Regression, KNN, and SVM so that large-valued measurements such as Area do not dominate smaller-valued shape measurements simply because of their units. The scaler was fitted only on training data and saved with each model in a pipeline. Decision Tree and Gaussian Naive Bayes were trained without scaling.
- **Decision Tree complexity:** The tree uses a maximum depth of 12 and requires at least 5 samples to split a node. These settings limit tree complexity and were chosen to reduce overfitting. They are implementation choices, not values established as optimal through a documented hyperparameter search.
- **KNN neighbors and weighting:** KNN uses five neighbors with Euclidean distance after scaling. Distance weighting gives closer neighbors more influence than farther neighbors. Five neighbors was the selected setting; the project does not claim that it was the best value from a tuning study.
- **SVM probability calibration:** The RBF SVM produces decision scores. Sigmoid calibration with five cross-validation folds within the training data converts these scores into class probability estimates used for multiclass AUC. With ensemble=False, the final classifier is fitted on the full training partition. The held-out test data is not used to fit the scaler, classifier, or calibrator. Calibration does not by itself guarantee better accuracy or perfectly calibrated probabilities.

## 5. Model Performance Comparison

All five classifiers were evaluated using the same
held-out test dataset containing 2,709 samples.

The following six evaluation metrics were calculated:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

For multiclass classification, macro averaging was used
for Precision, Recall, F1 Score, and one-vs-rest AUC.

### 5.1 Evaluation Results

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9192 | 0.9947 | 0.9307 | 0.9300 | 0.9302 | 0.9023 |
| Decision Tree | 0.9040 | 0.9659 | 0.9190 | 0.9180 | 0.9184 | 0.8839 |
| KNN | 0.9169 | 0.9834 | 0.9301 | 0.9267 | 0.9281 | 0.8996 |
| Gaussian Naive Bayes | 0.7630 | 0.9670 | 0.7736 | 0.7685 | 0.7669 | 0.7143 |
| Support Vector Machine | 0.9210 | 0.9907 | 0.9342 | 0.9317 | 0.9328 | 0.9045 |

### 5.2 Evaluation Method

1. All models were trained using the same 10,834 training samples.
2. All models were evaluated using the same 2,709 test samples.
3. The test dataset was kept separate from model training.
4. Previously saved models were loaded using joblib.
5. Class labels and probabilities were predicted on the test data.
6. All six metrics were calculated using scikit-learn.
7. Models were not retrained during evaluation.

The evaluation results are saved in:

`results/model_comparison.csv`

## 6. Observations on Individual Models

The following observations are based on the evaluation of
all five classifiers using the same 2,709 test samples.

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved 91.92% accuracy and the highest AUC (0.9947). Its F1 score of 0.9302 indicates strong classification performance across the seven bean varieties. |
| Decision Tree | Achieved 90.40% accuracy with an F1 score of 0.9184. Its performance was lower than Logistic Regression, KNN, and SVM. |
| K-Nearest Neighbors | Achieved 91.69% accuracy and an F1 score of 0.9281. Feature scaling helped ensure that the Euclidean distance calculation was not dominated by features with larger numerical ranges. |
| Gaussian Naive Bayes | Achieved the lowest accuracy (76.30%) and F1 score (0.7669). Its conditional-independence and Gaussian feature assumptions may limit performance on this dataset. However, its AUC of 0.9670 indicates relatively strong probability-based discrimination. |
| Support Vector Machine | Achieved the highest accuracy (92.10%), Precision (0.9342), Recall (0.9317), F1 (0.9328), and MCC (0.9045). The RBF kernel enables nonlinear classification. |

### Overall Observation

Support Vector Machine achieved the best overall performance
based on Accuracy, Precision, Recall, F1 Score, and MCC.

Logistic Regression achieved the highest AUC score (0.9947),
compared with SVM (0.9907).

The performance difference between SVM and Logistic Regression
was relatively small, with an accuracy difference of
0.18 percentage points.

Therefore, SVM is selected as the best-performing classifier
for this dataset based on the selected evaluation metrics.

These observations describe performance on the held-out test
set and do not establish statistical significance.

## 7. Overall Best Performing Model

### Selected Model: Support Vector Machine (SVM)

Based on the experimental results, Support Vector Machine
was selected as the best-performing classifier for the
Dry Bean Dataset.

### Performance of the Selected Model

| Evaluation Metric | SVM Score |
|---|---:|
| Accuracy | 0.9210 |
| AUC | 0.9907 |
| Precision | 0.9342 |
| Recall | 0.9317 |
| F1 Score | 0.9328 |
| MCC | 0.9045 |

### Justification

SVM achieved the highest scores among the five classifiers
in five of the six evaluation metrics:

- Accuracy: 92.10%
- Precision: 93.42%
- Recall: 93.17%
- F1 Score: 93.28%
- MCC: 0.9045

Logistic Regression achieved the highest AUC of 0.9947,
compared with SVM's AUC of 0.9907.

Although the performance difference between these two
models is small, SVM demonstrated the strongest overall
results on the held-out test dataset.

Therefore, SVM is selected as the winning classification
model for this experiment.

This conclusion is based on the observed test-set metrics
and does not imply that SVM will always outperform the
other classifiers on different datasets.
