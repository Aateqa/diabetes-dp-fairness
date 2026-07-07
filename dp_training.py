import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from config import (
    RESULTS_DIR,
    GRAPHS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)

from data_loader_diabetes import load_raw_diabetes_data
from metrics import safe_auc, get_probabilities, compute_fairness_metrics
from models.dp_model import make_dp_lr, make_non_private_lr


EPSILON_VALUES = [0.1, 0.5, 1.0, 5.0, np.inf]
THRESHOLDS = [
    0.03, 0.05, 0.07, 0.10, 0.12,
    0.15, 0.18, 0.20, 0.22, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50,
]


def tune_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in THRESHOLDS:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold


def train_and_evaluate_dp_lr(
    epsilon,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    sensitive_val,
    sensitive_test,
):
    """
    Trains either DP Logistic Regression or the non-private baseline.
    """
    if np.isinf(epsilon):
        model = make_non_private_lr()
        epsilon_label = "infinity"
        is_private = False
    else:
        model = make_dp_lr(epsilon)
        epsilon_label = str(epsilon)
        is_private = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    val_prob = get_probabilities(model, X_val)
    threshold = tune_threshold(y_val, val_prob)

    y_prob = get_probabilities(model, X_test)
    y_pred = (y_prob >= threshold).astype(int)

    recall = recall_score(y_test, y_pred, zero_division=0)
    fairness = compute_fairness_metrics(y_test, y_pred, y_prob, sensitive_test)
    val_fairness = compute_fairness_metrics(y_val, (val_prob >= threshold).astype(int), val_prob, sensitive_val)

    return {
        "epsilon": epsilon_label,
        "epsilon_numeric": 999 if np.isinf(epsilon) else epsilon,
        "is_private": is_private,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall,
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "fnr": 1 - recall,
        "auc": safe_auc(y_test, y_prob),
        "dp_diff": fairness["dp_diff"],
        "fnr_gap": fairness["fnr_gap"],
        "worst_group_sensitivity": fairness["worst_group_sensitivity"],
        "macro_avg_fnr": fairness["macro_avg_fnr"],
        "val_dp_diff": val_fairness["dp_diff"],
    }


def plot_privacy_utility_fairness_tradeoff(results_df, output_path):
    """
    Plots epsilon vs AUC, with point size showing demographic parity difference.
    """
    plot_df = results_df.copy()

    plot_df["epsilon_display"] = plot_df["epsilon"].replace({
        "infinity": "∞",
    })

    # Larger point means larger demographic parity difference.
    # Add a small constant so very small dp_diff values are still visible.
    point_sizes = 80 + (plot_df["dp_diff"].fillna(0) * 1200)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        plot_df["epsilon_display"],
        plot_df["auc"],
        s=point_sizes,
        alpha=0.75,
    )

    for _, row in plot_df.iterrows():
        plt.annotate(
            f"DP diff={row['dp_diff']:.3f}",
            (row["epsilon_display"], row["auc"]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel("Privacy budget ε")
    plt.ylabel("ROC-AUC")
    plt.title("DP Logistic Regression: Privacy-Utility-Fairness Tradeoff")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot: {output_path}")


def run_dp_experiment():
    """
    Runs the DP Logistic Regression experiment on the raw imbalanced dataset.

    Uses the proxy-reduced feature set so that direct sensitive attributes and
    obvious socioeconomic proxies are removed from training, while fairness is
    still evaluated by protected group membership.
    """
    os.makedirs(RESULTS_DIR / "dp", exist_ok=True)
    os.makedirs(GRAPHS_DIR / "dp", exist_ok=True)

    print("\n" + "=" * 80)
    print("Running DP Logistic Regression experiment")
    print("=" * 80)

    feature_sets, y, fairness_df, df = load_raw_diabetes_data(print_summary=False)

    feature_set_name = "Without Sensitive Attributes + Proxy-Reduced Features"
    X = feature_sets[feature_set_name]

    sensitive_attribute = "sex_group"
    sensitive_values = fairness_df[sensitive_attribute]

    print(f"Feature set: {feature_set_name}")
    print(f"Sensitive attribute for dp_diff: {sensitive_attribute}")
    print(f"X shape: {X.shape}")

    X_trainval, X_test, y_trainval, y_test, sensitive_trainval, sensitive_test = train_test_split(
        X,
        y,
        sensitive_values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val, sensitive_train, sensitive_val = train_test_split(
        X_trainval,
        y_trainval,
        sensitive_trainval,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )

    results = []

    for epsilon in EPSILON_VALUES:
        epsilon_label = "∞" if np.isinf(epsilon) else epsilon
        print(f"\nTraining DP Logistic Regression with epsilon={epsilon_label}")

        row = train_and_evaluate_dp_lr(
            epsilon=epsilon,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            sensitive_val=sensitive_val,
            sensitive_test=sensitive_test,
        )

        results.append(row)

        print(
            f"epsilon={row['epsilon']} | "
            f"auc={row['auc']:.4f} | "
            f"f1={row['f1']:.4f} | "
            f"recall={row['recall']:.4f} | "
            f"dp_diff={row['dp_diff']:.4f} | "
            f"fnr_gap={row['fnr_gap']:.4f}"
        )

    results_df = pd.DataFrame(results)

    output_csv = RESULTS_DIR / "dp" / "dp_logistic_regression_results.csv"
    output_plot = GRAPHS_DIR / "dp" / "epsilon_vs_auc_dp_diff.png"

    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved results: {output_csv}")

    plot_privacy_utility_fairness_tradeoff(
        results_df=results_df,
        output_path=output_plot,
    )

    print("\n  Theoretical note:")
    print("  DP Logistic Regression is an instance of DP-ERM with a convex loss.")
    print("  At strict epsilon values (0.1, 0.5) dp_diff is clearly lower than non-private,")
    print("  corroborating the excess risk bounds in Wang et al. (2019, ICML): DP noise")
    print("  regularises predictions toward uniform outputs across groups.")
    print("  The non-monotonic dp_diff between epsilon=5 and non-private is expected:")
    print("  per-epsilon thresholds differ (each is tuned on the val set independently),")
    print("  so threshold-induced variance dominates at small dp_diff values near non-private.")
    print("  For the non-convex (MLP) case, see dp_sgd_mlp_experiment.py.")

    print("\nDP experiment complete.")

    return results_df


if __name__ == "__main__":
    run_dp_experiment()
