"""
transfer_experiment.py

Subgroup generalisation experiment: does a model trained on high-income patients
transfer to low-income patients, and can importance weighting close the gap?

Research question:
    Healthcare datasets are often biased toward higher-income, better-insured
    populations. If we deploy a model trained on such data to underserved groups,
    how much performance do we lose - and can covariate shift correction help?

Methods compared
----------------
1. Oracle            - train and test within the same target subgroup (upper bound)
2. Naive transfer    - train on source (high-income), test on target (low-income)
3. IPS transfer      - inverse probability / importance-weighted training on source
                       (Shimodaira 2000; standard covariate shift correction)

The gap between Naive and Oracle quantifies the generalisation failure.
The gap between IPS and Naive quantifies what classical adaptation buys.
Closing the remaining gap motivates causality-based invariant learning.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split

from config import RESULTS_DIR, GRAPHS_DIR, RANDOM_STATE, TEST_SIZE
from data_loader_diabetes import load_raw_diabetes_data
from metrics import safe_auc, get_probabilities, compute_fairness_metrics
from models.tree_ensemble import make_xgb_model


SOURCE_GROUP = "high_income"
TARGET_GROUP = "low_income"
FEATURE_SET = "Without Sensitive Attributes"


def estimate_importance_weights(X_source, X_target):
    """
    Estimate importance weights w(x) = P(target|x) / P(source|x).

    A logistic regression domain classifier is trained to distinguish source
    from target samples. The predicted probabilities define the density ratio.
    Weights are clipped to [0.05, 0.95] and normalised to mean=1.
    """
    X_all = np.vstack([X_source, X_target])
    domain = np.array([0] * len(X_source) + [1] * len(X_target))

    scaler = StandardScaler()
    X_all_sc = scaler.fit_transform(X_all)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
    clf.fit(X_all_sc, domain)

    p_target = clf.predict_proba(scaler.transform(X_source))[:, 1]
    p_target = np.clip(p_target, 0.05, 0.95)

    weights = p_target / (1.0 - p_target)
    weights = weights / weights.mean()

    return weights


def evaluate(model, X_test, y_test, sex_test):
    y_pred = model.predict(X_test)
    y_prob = get_probabilities(model, X_test)
    recall = recall_score(y_test, y_pred, zero_division=0)
    fairness = compute_fairness_metrics(y_test, y_pred, y_prob, sex_test)

    return {
        "auc": safe_auc(y_test, y_prob),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "recall": recall,
        "fnr": 1 - recall,
        "dp_diff": fairness["dp_diff"],
        "fnr_gap": fairness["fnr_gap"],
        "worst_group_sensitivity": fairness["worst_group_sensitivity"],
    }


def run_transfer_experiment():
    os.makedirs(RESULTS_DIR / "transfer", exist_ok=True)
    os.makedirs(GRAPHS_DIR / "transfer", exist_ok=True)

    print("\n" + "=" * 80)
    print("Transfer learning / subgroup generalisation experiment")
    print(f"  Source domain : {SOURCE_GROUP}")
    print(f"  Target domain : {TARGET_GROUP}")
    print(f"  Feature set   : {FEATURE_SET}")
    print("=" * 80)

    feature_sets, y, fairness_df, _ = load_raw_diabetes_data(print_summary=False)
    X = feature_sets[FEATURE_SET].copy()

    income_groups = fairness_df["income_group"]
    sex_groups = fairness_df["sex_group"]

    src_mask = income_groups == SOURCE_GROUP
    tgt_mask = income_groups == TARGET_GROUP

    X_src, y_src = X[src_mask].copy(), y[src_mask].copy()
    X_tgt, y_tgt = X[tgt_mask].copy(), y[tgt_mask].copy()
    sex_tgt = sex_groups[tgt_mask].copy()

    print(f"\nSource  ({SOURCE_GROUP}): n={len(X_src):,}  "
          f"diabetes rate={y_src.mean():.3f}")
    print(f"Target  ({TARGET_GROUP}): n={len(X_tgt):,}  "
          f"diabetes rate={y_tgt.mean():.3f}")

    # Hold out test set from the target domain.
    (X_tgt_train, X_tgt_test,
     y_tgt_train, y_tgt_test,
     sex_tgt_train, sex_tgt_test) = train_test_split(
        X_tgt, y_tgt, sex_tgt,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_tgt,
    )

    results = []

    # --- 1. Oracle ---
    print("\n[1/3] Oracle: train on target, test on target")
    oracle = make_xgb_model(random_state=RANDOM_STATE)
    oracle.fit(X_tgt_train, y_tgt_train)
    m = evaluate(oracle, X_tgt_test, y_tgt_test, sex_tgt_test)
    m["method"] = "Oracle (train on target)"
    results.append(m)
    print(f"      AUC={m['auc']:.4f}  F1={m['f1']:.4f}  "
          f"dp_diff={m['dp_diff']:.4f}  fnr_gap={m['fnr_gap']:.4f}")

    # --- 2. Naive transfer ---
    print(f"\n[2/3] Naive transfer: train on {SOURCE_GROUP}, test on {TARGET_GROUP}")
    naive = make_xgb_model(random_state=RANDOM_STATE)
    naive.fit(X_src, y_src)
    m = evaluate(naive, X_tgt_test, y_tgt_test, sex_tgt_test)
    m["method"] = "Naive transfer"
    results.append(m)
    print(f"      AUC={m['auc']:.4f}  F1={m['f1']:.4f}  "
          f"dp_diff={m['dp_diff']:.4f}  fnr_gap={m['fnr_gap']:.4f}")

    # --- 3. Importance-weighted transfer ---
    print("\n[3/3] Importance-weighted (IPS) transfer")
    weights = estimate_importance_weights(X_src.values, X_tgt_train.values)
    ips = make_xgb_model(random_state=RANDOM_STATE)
    ips.fit(X_src, y_src, sample_weight=weights)
    m = evaluate(ips, X_tgt_test, y_tgt_test, sex_tgt_test)
    m["method"] = "IPS-weighted transfer"
    results.append(m)
    print(f"      AUC={m['auc']:.4f}  F1={m['f1']:.4f}  "
          f"dp_diff={m['dp_diff']:.4f}  fnr_gap={m['fnr_gap']:.4f}")

    results_df = pd.DataFrame(results)

    csv_path = RESULTS_DIR / "transfer" / "transfer_experiment_results.csv"
    results_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("Summary")
    print(results_df.round(4).to_string(index=False))
    print("=" * 80)

    _auc_gap = results_df.loc[results_df["method"] == "Naive transfer", "auc"].values[0]
    _auc_oracle = results_df.loc[results_df["method"] == "Oracle (train on target)", "auc"].values[0]
    _auc_ips = results_df.loc[results_df["method"] == "IPS-weighted transfer", "auc"].values[0]

    print(f"\n  Naive vs Oracle AUC gap  : {_auc_oracle - _auc_gap:.4f}")
    print(f"  IPS vs Naive AUC gain    : {_auc_ips - _auc_gap:.4f}")
    print(f"  Remaining gap (IPS→Oracle): {_auc_oracle - _auc_ips:.4f}")
    print("  (Remaining gap motivates causality-based invariant learning.)")

    plot_transfer_comparison(results_df, GRAPHS_DIR / "transfer" / "transfer_comparison.png")

    print(f"\nSaved: {csv_path}")
    print("\nTransfer experiment complete.")

    return results_df


def plot_transfer_comparison(results_df, output_path):
    metrics = [
        ("auc", "ROC-AUC"),
        ("f1", "F1-score"),
        ("recall", "Recall"),
        ("fnr", "FNR"),
        ("dp_diff", "DP Diff (Sex)"),
        ("worst_group_sensitivity", "Worst-Group Sens."),
    ]

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5))

    colors = ["forestgreen", "tomato", "steelblue"]
    methods = results_df["method"].tolist()

    for ax, (col, label) in zip(axes, metrics):
        values = results_df[col].tolist()
        bars = ax.bar(range(len(methods)), values, color=colors[: len(methods)])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=7)
        ax.set_title(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

    plt.suptitle(
        f"Transfer Learning: {SOURCE_GROUP} → {TARGET_GROUP}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run_transfer_experiment()
