# Lab 1 Report: Environment and First System Measurements

## 1. Goal
The goal of this lab is to establish an isolated, reproducible Python environment, evaluate baseline machine learning models on the Breast Cancer Wisconsin dataset, measure critical system metrics (training duration, single-sample inference latency, serialized model size, and peak memory usage), and assess deployment fit across target computing tiers.

## 2. Method
- **Environment**: Isolated Python 3.11 environment with exact dependency versions pinned in `requirements.txt`.
- **Dataset & Split**: Breast Cancer Wisconsin dataset loaded via `sklearn.datasets.load_breast_cancer`. Split into 70% training and 30% testing subsets with stratified sampling using `random_state=42`.
- **Models**:
  1. Logistic Regression (`LogisticRegression(max_iter=1000, random_state=42)`)
  2. Random Forest (`RandomForestClassifier(n_estimators=100, random_state=42)`)
- **Measurement Protocol**:
  - Training time: Median of 5 runs after 1 initial warm-up run.
  - Inference latency: Median single-sample inference time over 100 repetitions.
  - Model size: Serialization using `joblib.dump` to file.
  - Memory usage: Peak Resident Set Size (RSS) captured via `psutil`.

## 3. Results
All generated outputs and metrics were successfully computed and stored in `lab01/results/`:
- `versions.txt`: Full environment package versions.
- `baseline_accuracy.csv`: Evaluation accuracy scores.
- `logisticregression_model.joblib` and `randomforest_model.joblib`: Serialized model artifacts.

## 4. Conclusions
1. Both models achieve strong test accuracy, but Logistic Regression offers faster execution and lower memory usage compared to Random Forest.
2. Both models satisfy resource budgets for Cloud, Edge, and Mobile deployment tiers.
3. Standard Random Forest ensembles exceed the strict hardware limits of TinyML devices ($\le 256\text{ KB}$ RAM, $\le 100\text{ KB}$ flash storage).
