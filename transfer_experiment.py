"""
transfer_experiment.py

Subgroup generalisation experiment with:
1. target-domain threshold tuning for screening sensitivity
2. stronger IPS/domain-ratio diagnostics for covariate-shift correction
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import RESULTS_DIR, GRAPHS_DIR, RANDOM_STATE, TEST_SIZE
from data_loader_diabetes import load_raw_diabetes_data
from metrics import safe_auc, get_probabilities, compute_fairness_metrics
from models.tree_ensemble import make_xgb_model


SOURCE_GROUP = "high_income"
TARGET_GROUP = "low_income"
FEATURE_SET = "Without Sensitive Attributes + Proxy-Reduced Features"


def estimate_importance_weights(X_source, X_target):
    """
    Estimate source-sample importance weights for source -> target adaptation.

    A domain classifier predicts whether x looks like target-domain data.
    The density ratio is approximated using odds P(target|x) / P(source|x).
    We train the domain classifier with balanced domain weights so the ratio
    is not dominated by the larger source sample size.
    """
    X_all = np.vstack([X_source, X_target])
    domain = np.array([0] * len(X_source) + [1] * len(X_target))

    X_dom_train, X_dom_val, y_dom_train, y_dom_val = train_test_split(
        X_all,
        domain,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=domain,
    )

    scaler = StandardScaler()
    X_dom_train_sc = scaler.fit_transform(X_dom_train)
    X_dom_val_sc = scaler.transform(X_dom_val)
    X_source_sc = scaler.transform(X_source)

    # Balance source/target domain labels during domain-classifier training.
    n_source_train = np.sum(y_dom_train == 0)
    n_target_train = np.sum(y_dom_train == 1)
    sample_weight = np.where(
        y_dom_train == 1,
        0.5 / max(n_target_train, 1),
        0.5 / max(n_source_train, 1),
    )
    sample_weight = sample_weight * len(sample_weight)

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_dom_train_sc, y_dom_train, sample_weight=sample_weight)

    dom_val_prob = clf.predict_proba(X_dom_val_sc)[:, 1]
    domain_auc = safe_auc(y_dom_val, dom_val_prob)

    p_target = clf.predict_proba(X_source_sc)[:, 1]
    p_target = np.clip(p_target, 0.02, 0.98)

    raw_weights = p_target / (1.0 - p_target)

    # Robust clipping avoids a few extreme source samples dominating training.
    clipped_weights = np.clip(raw_weights, 0.1, 10.0)
    weights = clipped_weights / clipped_weights.mean()

    diagnostics = {
        "domain_auc": float(domain_auc),
        "p_target_min": float(p_target.min()),
        "p_target_median": float(np.median(p_target)),
        "p_target_max": float(p_target.max()),
        "raw_weight_min": float(raw_weights.min()),
        "raw_weight_median": float(np.median(raw_weights)),
        "raw_weight_max": float(raw_weights.max()),
        "weight_min": float(weights.min()),
        "weight_median": float(np.median(weights)),
        "weight_max": float(weights.max()),
        "weight_std": float(weights.std()),
    }

    return weights, diagnostics


def tune_threshold(y_val, y_prob, sensitive_features, min_threshold=0.05, max_threshold=0.75):
    """
    Tune threshold for diabetes screening without collapsing to an extreme
    low threshold.

    We first search for thresholds that keep recall high, precision usable,
    and FNR gap controlled. If no threshold satisfies the constraints, we fall
    back to the best screening-oriented score.
    """
    thresholds = np.linspace(min_threshold, max_threshold, 71)
    rows = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        fairness = compute_fairness_metrics(y_val, y_pred, y_prob, sensitive_features)

        worst_group_sensitivity = fairness["worst_group_sensitivity"]
        macro_avg_fnr = fairness["macro_avg_fnr"]
        fnr_gap = fairness["fnr_gap"]

        # Balanced screening score: still prioritises sensitivity, but avoids
        # selecting a threshold that creates excessive false positives.
        score = (
            1.20 * worst_group_sensitivity
            + 0.50 * f1
            + 0.25 * precision
            - 0.50 * macro_avg_fnr
            - 0.75 * fnr_gap
        )

        rows.append({
            "threshold": float(threshold),
            "score": float(score),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fnr": float(1 - recall),
            "dp_diff": float(fairness["dp_diff"]),
            "fnr_gap": float(fnr_gap),
            "worst_group_sensitivity": float(worst_group_sensitivity),
            "macro_avg_fnr": float(macro_avg_fnr),
        })

    df = pd.DataFrame(rows)

    feasible = df[
        (df["recall"] >= 0.75)
        & (df["precision"] >= 0.30)
        & (df["fnr_gap"] <= 0.05)
    ]

    if len(feasible) > 0:
        best = feasible.sort_values(
            ["score", "worst_group_sensitivity", "f1"],
            ascending=[False, False, False],
        ).iloc[0]
    else:
        best = df.sort_values(
            ["score", "worst_group_sensitivity", "f1"],
            ascending=[False, False, False],
        ).iloc[0]

    return float(best["threshold"]), df


def evaluate(model, X_test, y_test, sex_test, threshold):
    y_prob = get_probabilities(model, X_test)
    y_pred = (y_prob >= threshold).astype(int)

    recall = recall_score(y_test, y_pred, zero_division=0)
    fairness = compute_fairness_metrics(y_test, y_pred, y_prob, sex_test)

    return {
        "threshold": float(threshold),
        "auc": safe_auc(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "recall": recall,
        "fnr": 1 - recall,
        "dp_diff": fairness["dp_diff"],
        "fnr_gap": fairness["fnr_gap"],
        "worst_group_sensitivity": fairness["worst_group_sensitivity"],
        "macro_avg_fnr": fairness["macro_avg_fnr"],
    }


def run_transfer_experiment():
    os.makedirs(RESULTS_DIR / "transfer", exist_ok=True)
    os.makedirs(GRAPHS_DIR / "transfer", exist_ok=True)

    print("\n" + "=" * 80)
    print("Transfer learning / subgroup generalisation experiment")
    print(f"  Source domain : {SOURCE_GROUP}")
    print(f"  Target domain : {TARGET_GROUP}")
    print(f"  Feature set   : {FEATURE_SET}")
    print("  Thresholding  : tuned on target validation set")
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

    print(f"\nSource  ({SOURCE_GROUP}): n={len(X_src):,}  diabetes rate={y_src.mean():.3f}")
    print(f"Target  ({TARGET_GROUP}): n={len(X_tgt):,}  diabetes rate={y_tgt.mean():.3f}")

    # Split target into train / validation / test.
    X_tgt_temp, X_tgt_test, y_tgt_temp, y_tgt_test, sex_tgt_temp, sex_tgt_test = train_test_split(
        X_tgt,
        y_tgt,
        sex_tgt,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_tgt,
    )

    X_tgt_train, X_tgt_val, y_tgt_train, y_tgt_val, sex_tgt_train, sex_tgt_val = train_test_split(
        X_tgt_temp,
        y_tgt_temp,
        sex_tgt_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_tgt_temp,
    )

    results = []
    threshold_tables = []

    # --- 1. Oracle ---
    print("\n[1/3] Oracle: train on target, tune on target-val, test on target-test")
    oracle = make_xgb_model(random_state=RANDOM_STATE)
    oracle.fit(X_tgt_train, y_tgt_train)

    oracle_val_prob = get_probabilities(oracle, X_tgt_val)
    oracle_threshold, oracle_tuning = tune_threshold(y_tgt_val, oracle_val_prob, sex_tgt_val)
    oracle_tuning["method"] = "Oracle (train on target)"
    threshold_tables.append(oracle_tuning)

    m = evaluate(oracle, X_tgt_test, y_tgt_test, sex_tgt_test, oracle_threshold)
    m["method"] = "Oracle (train on target)"
    results.append(m)
    print(
        f"      threshold={oracle_threshold:.3f} | AUC={m['auc']:.4f} | "
        f"precision={m['precision']:.4f} | F1={m['f1']:.4f} | recall={m['recall']:.4f} | "
        f"worst_group_sens={m['worst_group_sensitivity']:.4f} | fnr_gap={m['fnr_gap']:.4f}"
    )

    # --- 2. Naive transfer ---
    print(f"\n[2/3] Naive transfer: train on {SOURCE_GROUP}, tune on target-val, test on {TARGET_GROUP}")
    naive = make_xgb_model(random_state=RANDOM_STATE)
    naive.fit(X_src, y_src)

    naive_val_prob = get_probabilities(naive, X_tgt_val)
    naive_threshold, naive_tuning = tune_threshold(y_tgt_val, naive_val_prob, sex_tgt_val)
    naive_tuning["method"] = "Naive transfer"
    threshold_tables.append(naive_tuning)

    m = evaluate(naive, X_tgt_test, y_tgt_test, sex_tgt_test, naive_threshold)
    m["method"] = "Naive transfer"
    results.append(m)
    print(
        f"      threshold={naive_threshold:.3f} | AUC={m['auc']:.4f} | "
        f"precision={m['precision']:.4f} | F1={m['f1']:.4f} | recall={m['recall']:.4f} | "
        f"worst_group_sens={m['worst_group_sensitivity']:.4f} | fnr_gap={m['fnr_gap']:.4f}"
    )

    # --- 3. Importance-weighted transfer ---
    print("\n[3/3] Importance-weighted transfer: train weighted source, tune on target-val")
    weights, weight_diag = estimate_importance_weights(X_src.values, X_tgt_train.values)

    print(
        "      Domain classifier / weight diagnostics:\n"
        f"        domain_auc={weight_diag['domain_auc']:.4f}\n"
        f"        p_target min/p50/max="
        f"{weight_diag['p_target_min']:.3f}/"
        f"{weight_diag['p_target_median']:.3f}/"
        f"{weight_diag['p_target_max']:.3f}\n"
        f"        normalized weight min/p50/max/std="
        f"{weight_diag['weight_min']:.3f}/"
        f"{weight_diag['weight_median']:.3f}/"
        f"{weight_diag['weight_max']:.3f}/"
        f"{weight_diag['weight_std']:.3f}"
    )

    ips = make_xgb_model(random_state=RANDOM_STATE)
    ips.fit(X_src, y_src, sample_weight=weights)

    ips_val_prob = get_probabilities(ips, X_tgt_val)
    ips_threshold, ips_tuning = tune_threshold(y_tgt_val, ips_val_prob, sex_tgt_val)
    ips_tuning["method"] = "IPS-weighted transfer"
    threshold_tables.append(ips_tuning)

    m = evaluate(ips, X_tgt_test, y_tgt_test, sex_tgt_test, ips_threshold)
    m["method"] = "IPS-weighted transfer"
    m.update(weight_diag)
    results.append(m)
    print(
        f"      threshold={ips_threshold:.3f} | AUC={m['auc']:.4f} | "
        f"precision={m['precision']:.4f} | F1={m['f1']:.4f} | recall={m['recall']:.4f} | "
        f"worst_group_sens={m['worst_group_sensitivity']:.4f} | fnr_gap={m['fnr_gap']:.4f}"
    )

    results_df = pd.DataFrame(results)
    tuning_df = pd.concat(threshold_tables, ignore_index=True)

    csv_path = RESULTS_DIR / "transfer" / "transfer_experiment_results.csv"
    tuning_path = RESULTS_DIR / "transfer" / "transfer_threshold_tuning.csv"

    results_df.to_csv(csv_path, index=False)
    tuning_df.to_csv(tuning_path, index=False)

    print("\n" + "=" * 80)
    print("Summary")
    print(results_df.round(4).to_string(index=False))
    print("=" * 80)

    auc_naive = results_df.loc[results_df["method"] == "Naive transfer", "auc"].values[0]
    auc_oracle = results_df.loc[results_df["method"] == "Oracle (train on target)", "auc"].values[0]
    auc_ips = results_df.loc[results_df["method"] == "IPS-weighted transfer", "auc"].values[0]

    print(f"\n  Naive vs Oracle AUC gap     : {auc_oracle - auc_naive:.4f}")
    print(f"  IPS vs Naive AUC gain       : {auc_ips - auc_naive:.4f}")
    print(f"  Remaining gap IPS->Oracle   : {auc_oracle - auc_ips:.4f}")
    print("  Threshold tuning targets screening sensitivity and worst-group sensitivity.")

    plot_transfer_comparison(results_df, GRAPHS_DIR / "transfer" / "transfer_comparison.png")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {tuning_path}")
    print("\nTransfer experiment complete.")

    return results_df


def plot_transfer_comparison(results_df, output_path):
    metrics = [
        ("auc", "ROC-AUC"),
        ("precision", "Precision"),
        ("f1", "F1-score"),
        ("recall", "Recall"),
        ("fnr", "FNR"),
        ("fnr_gap", "FNR Gap"),
        ("worst_group_sensitivity", "Worst-Group Sens."),
        ("macro_avg_fnr", "Macro Avg FNR"),
    ]

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 5))

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
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    plt.suptitle(
        f"Transfer Learning: {SOURCE_GROUP} -> {TARGET_GROUP} with threshold tuning",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run_transfer_experiment()
