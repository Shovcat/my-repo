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

| Metric | Logistic Regression | Random Forest |
| :--- | :--- | :--- |
| **Test Accuracy** | 0.9415 | 0.9357 |
| **Training Time (sec)** | 0.05529 | 0.07207 |
| **Single-Sample Latency (ms)** | 0.0640 | 2.4543 |
| **Model Size (KB)** | 1.08 | 284.10 |
| **Peak RAM (MB)** | 181.17 | 182.83 |

## 4. Conclusions
1. Both models achieve strong test accuracy, but Logistic Regression offers substantially faster execution and lower model storage footprint compared to Random Forest.
2. Both models easily satisfy resource budgets for Cloud, Edge, and Mobile deployment tiers.
3. Standard Random Forest ensembles exceed the strict hardware limits of TinyML devices ($\le 256\text{ KB}$ RAM, $\le 100\text{ KB}$ flash storage) due to model size (284.10 KB).
