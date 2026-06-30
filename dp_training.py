import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from diffprivlib.models import LogisticRegression as DPLogisticRegression

from config import (
    RESULTS_DIR,
    GRAPHS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)

from data_loader_diabetes import load_raw_diabetes_data


EPSILON_VALUES = [0.1, 0.5, 1.0, 5.0, np.inf]


def safe_auc(y_true, y_prob):
    """
    Safely computes ROC-AUC.
    Returns NaN if AUC cannot be computed.
    """
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return np.nan


def compute_dp_diff(y_pred, sensitive_values):
    """
    Computes demographic parity difference.

    DP difference = max group selection rate - min group selection rate.
    Selection rate = proportion predicted positive.
    """
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    sensitive_values = pd.Series(sensitive_values).reset_index(drop=True)

    selection_rates = []

    for group in sorted(sensitive_values.dropna().unique()):
        mask = sensitive_values == group
        group_selection_rate = y_pred[mask].mean()
        selection_rates.append(group_selection_rate)

    if len(selection_rates) == 0:
        return np.nan

    return max(selection_rates) - min(selection_rates)


def get_probabilities(model, X_test):
    """
    Gets positive-class probabilities.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    return model.predict(X_test)


def make_non_private_lr():
    """
    Non-private Logistic Regression baseline, used for epsilon = infinity.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )),
    ])


def make_dp_lr(epsilon):
    """
    Differentially private Logistic Regression.

    diffprivlib's LogisticRegression needs numeric features.
    We scale features before fitting the private model.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", DPLogisticRegression(
            epsilon=epsilon,
            data_norm=10.0,
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])


def train_and_evaluate_dp_lr(
    epsilon,
    X_train,
    X_test,
    y_train,
    y_test,
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

    y_pred = model.predict(X_test)
    y_prob = get_probabilities(model, X_test)

    accuracy = accuracy_score(y_test, y_pred)
    auc = safe_auc(y_test, y_prob)
    dp_diff = compute_dp_diff(y_pred, sensitive_test)

    return {
        "epsilon": epsilon_label,
        "epsilon_numeric": 999 if np.isinf(epsilon) else epsilon,
        "is_private": is_private,
        "accuracy": accuracy,
        "auc": auc,
        "dp_diff": dp_diff,
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
    plt.title("DP Logistic Regression: Privacy–Utility–Fairness Tradeoff")
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

    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X,
        y,
        sensitive_values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []

    for epsilon in EPSILON_VALUES:
        epsilon_label = "∞" if np.isinf(epsilon) else epsilon
        print(f"\nTraining DP Logistic Regression with epsilon={epsilon_label}")

        row = train_and_evaluate_dp_lr(
            epsilon=epsilon,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            sensitive_test=sensitive_test,
        )

        results.append(row)

        print(
            f"epsilon={row['epsilon']} | "
            f"accuracy={row['accuracy']:.4f} | "
            f"auc={row['auc']:.4f} | "
            f"dp_diff={row['dp_diff']:.4f}"
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

    print("\nDP experiment complete.")

    return results_df


if __name__ == "__main__":
    run_dp_experiment()