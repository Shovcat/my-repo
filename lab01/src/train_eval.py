import os
import time
import random
import psutil
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Set global seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def get_peak_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def measure_training(model_cls, kwargs, X, y):
    # Warm-up run
    model = model_cls(**kwargs)
    model.fit(X, y)

    times = []
    memories = []
    for _ in range(5):
        m = model_cls(**kwargs)
        t0 = time.perf_counter()
        m.fit(X, y)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        memories.append(get_peak_memory_mb())

    median_time = np.median(times)
    peak_mem = np.max(memories)
    return model, median_time, peak_mem

def measure_inference_latency(model, sample):
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        model.predict(sample)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
    return np.median(latencies)

def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )

    models_spec = {
        "LogisticRegression": (LogisticRegression, {"max_iter": 1000, "random_state": SEED}),
        "RandomForest": (RandomForestClassifier, {"n_estimators": 100, "random_state": SEED})
    }

    accuracy_records = []
    metrics_summary = []

    os.makedirs("lab01/results", exist_ok=True)
    single_sample = X_test[0:1]

    for name, (cls, kwargs) in models_spec.items():
        trained_model, train_time, peak_mem = measure_training(cls, kwargs, X_train, y_train)
        acc = trained_model.score(X_test, y_test)
        latency_ms = measure_inference_latency(trained_model, single_sample)

        # Model serialization
        file_path = f"lab01/results/{name.lower()}_model.joblib"
        joblib.dump(trained_model, file_path)
        size_bytes = os.path.getsize(file_path)
        size_kb = size_bytes / 1024.0

        accuracy_records.append({"model": name, "accuracy": f"{acc:.4f}"})
        metrics_summary.append({
            "model": name,
            "accuracy": round(acc, 4),
            "train_time_sec": round(train_time, 5),
            "latency_ms": round(latency_ms, 4),
            "size_bytes": size_bytes,
            "size_kb": round(size_kb, 2),
            "peak_memory_mb": round(peak_mem, 2)
        })

    # Save accuracy CSV
    df_acc = pd.DataFrame(accuracy_records)
    df_acc.to_csv("lab01/results/baseline_accuracy.csv", index=False)

    print("--- Benchmark Summary ---")
    df_summary = pd.DataFrame(metrics_summary)
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()
