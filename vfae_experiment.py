import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)

from config import RESULTS_DIR, GRAPHS_DIR, RANDOM_STATE, TEST_SIZE
from data_loader_diabetes import load_raw_diabetes_data
from metrics import compute_fairness_metrics, compute_group_rates
from models.vfae import VFAEClassifier


FEATURE_SET_NAME = "Without Sensitive Attributes + Proxy-Reduced Features"
SENSITIVE_ATTRIBUTE = "sex_group"


def safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return np.nan


def safe_brier(y_true, y_prob):
    try:
        return brier_score_loss(y_true, y_prob)
    except ValueError:
        return np.nan


def run_vfae_experiment():
    os.makedirs(RESULTS_DIR / "vfae", exist_ok=True)
    os.makedirs(GRAPHS_DIR / "vfae", exist_ok=True)

    print("\n" + "=" * 80)
    print("Running standalone VFAE experiment")
    print("=" * 80)

    feature_sets, y, fairness_df, full_df = load_raw_diabetes_data(print_summary=False)

    if FEATURE_SET_NAME not in feature_sets:
        print("Available feature sets:")
        for key in feature_sets.keys():
            print(f"  - {key}")
        raise KeyError(f"Feature set not found: {FEATURE_SET_NAME}")

    X = feature_sets[FEATURE_SET_NAME].copy()
    y = y.astype(int).copy()
    sensitive = fairness_df[SENSITIVE_ATTRIBUTE].copy()

    print(f"Feature set: {FEATURE_SET_NAME}")
    print(f"Sensitive attribute: {SENSITIVE_ATTRIBUTE}")
    print(f"X shape: {X.shape}")

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X,
        y,
        sensitive,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = VFAEClassifier(
        n_epochs=20,
        batch_size=1024,
        random_state=RANDOM_STATE,
    )

    print("\nTraining VFAEClassifier...")
    model.fit(
        X_train,
        y_train,
        sensitive_features=s_train,
    )

    print("\nEvaluating VFAEClassifier...")

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "model": "VFAEClassifier",
        "feature_set": FEATURE_SET_NAME,
        "threshold": getattr(model, "_threshold", np.nan),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "fnr": 1 - recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": safe_auc(y_test, y_prob),
        "brier": safe_brier(y_test, y_prob),
    }

    group_df = compute_group_rates(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        group_values=s_test,
    )

    if not group_df.empty:
        metrics.update(compute_fairness_metrics(y_test, y_pred, y_prob, s_test))

    metrics_df = pd.DataFrame([metrics])

    metrics_path = RESULTS_DIR / "vfae" / "vfae_standalone_metrics.csv"
    group_path = RESULTS_DIR / "vfae" / "vfae_standalone_group_metrics.csv"
    pred_path = RESULTS_DIR / "vfae" / "vfae_standalone_predictions.csv"
    plot_path = GRAPHS_DIR / "vfae" / "vfae_standalone_metrics.png"

    metrics_df.to_csv(metrics_path, index=False)

    if not group_df.empty:
        group_df.to_csv(group_path, index=False)

    predictions_df = pd.DataFrame({
        "y_true": np.asarray(y_test),
        "sensitive": np.asarray(s_test),
        "y_prob": y_prob,
        "y_pred": y_pred,
    })
    predictions_df.to_csv(pred_path, index=False)

    print("\nVFAE standalone metrics:")
    print(metrics_df.round(4).T)

    selected_cols = [
        col for col in [
            "auc",
            "recall",
            "fnr",
            "f1",
            "precision",
            "accuracy",
            "worst_group_sensitivity",
            "macro_avg_fnr",
            "fnr_gap",
            "dp_diff",
        ]
        if col in metrics_df.columns
    ]

    if selected_cols:
        plot_df = metrics_df[selected_cols].T
        plot_df.columns = ["value"]

        plt.figure(figsize=(10, 6))
        plt.bar(plot_df.index, plot_df["value"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Metric value")
        plt.title("Standalone VFAE Metrics")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Saved plot: {plot_path}")

    print("\nSaved:")
    print(metrics_path)
    if not group_df.empty:
        print(group_path)
    print(pred_path)

    print("\nStandalone VFAE experiment complete.")

    return metrics_df


if __name__ == "__main__":
    run_vfae_experiment()
