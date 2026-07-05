"""
compare_all_results.py

Loads outputs from all experiment types and produces a unified comparison table and plots.

Run after:
    python main.py
    python dp_training.py
    python vfae_experiment.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR, GRAPHS_DIR


METHOD_COLORS = {
    "Standard": "steelblue",
    "Fairness-Aware": "mediumseagreen",
    "Differential Privacy": "darkorange",
    "Fair Deep Representation": "mediumpurple",
    "DP-SGD Deep Learning": "sienna",
    "Invariant Learning": "teal",
    "Transfer Learning": "goldenrod",
    "Transfer/Oracle": "slategray",
    "Privacy Attack": "crimson",
}


def with_comparison_group(row, comparison_group):
    row["comparison_group"] = comparison_group
    row["is_directly_comparable"] = comparison_group != "privacy_audit"
    return row


def load_cv_results():
    """
    Loads per-model mean metrics from the main 5-fold CV experiment.
    Uses the raw dataset and keeps all three feature sets:
    Original Features, Without Sensitive Attributes, and Proxy-Reduced Features.
    """
    path = RESULTS_DIR / "raw_dataset" / "final_cv_results_all_models.csv"
    if not path.exists():
        print(f"  CV results not found: {path}  ->  run main.py first.")
        return None

    df = pd.read_csv(path)

    fairness_model = "Fairlearn-DP"
    rows = []
    for _, row in df.iterrows():
        if row["model"] == fairness_model:
            method_type = "Fairness-Aware"
        elif row["model"] == "VFAE":
            method_type = "Fair Deep Representation"
        else:
            method_type = "Standard"
        rows.append(with_comparison_group({
            "model": row["model"],
            "feature_set": row.get("feature_set"),
            "method_type": method_type,
            "eval_protocol": "5-fold CV",
            "auc": row.get("auc_mean"),
            "f1": row.get("f1_mean"),
            "recall": row.get("recall_mean"),
            "fnr": row.get("fnr_mean"),
            "dp_diff": row.get("sex_group_dp_diff_mean"),
            "fnr_gap": row.get("sex_group_fnr_gap_mean"),
            "worst_group_sensitivity": row.get("sex_group_worst_group_sensitivity_mean"),
            "macro_avg_fnr": row.get("sex_group_macro_avg_fnr_mean"),
        }, "cv_raw"))

    return pd.DataFrame(rows)


def load_dp_results():
    """Loads DP Logistic Regression results across epsilon values."""
    path = RESULTS_DIR / "dp" / "dp_logistic_regression_results.csv"
    if not path.exists():
        print(f"  DP results not found: {path}  ->  run dp_training.py first.")
        return None

    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        eps = row["epsilon"]
        rows.append(with_comparison_group({
            "model": f"DP-LR (ε={eps})",
            "feature_set": "Without Sensitive Attributes + Proxy-Reduced Features",
            "method_type": "Differential Privacy",
            "eval_protocol": "Holdout",
            "auc": row.get("auc"),
            "f1": row.get("f1"),
            "recall": row.get("recall"),
            "fnr": row.get("fnr"),
            "dp_diff": row.get("dp_diff"),
            "fnr_gap": row.get("fnr_gap"),
            "worst_group_sensitivity": row.get("worst_group_sensitivity"),
            "macro_avg_fnr": row.get("macro_avg_fnr"),
        }, "holdout_raw"))

    return pd.DataFrame(rows)


def load_dp_sgd_mlp_results():
    """Loads Opacus DP-SGD MLP results across epsilon values."""
    path = RESULTS_DIR / "dp_sgd_mlp" / "dp_sgd_mlp_results.csv"
    if not path.exists():
        print(f"  DP-SGD MLP results not found: {path}  ->  run dp_sgd_mlp_experiment.py first.")
        return None

    df = pd.read_csv(path)
    rows = []

    for _, row in df.iterrows():
        method = row.get("method", "DP-SGD MLP")
        clipping = row.get("clipping", "fixed")

        if method == "MLP non-private":
            model_name = "MLP non-private"
            method_type = "Standard"
        else:
            target_eps = row.get("target_epsilon", np.nan)
            clip_tag = f" [{clipping}]" if pd.notna(clipping) and clipping != "none" else ""
            model_name = f"DP-SGD MLP (ε={target_eps}){clip_tag}" if pd.notna(target_eps) else method
            method_type = "DP-SGD Deep Learning"

        rows.append(with_comparison_group({
            "model": model_name,
            "feature_set": "Without Sensitive Attributes + Proxy-Reduced Features",
            "method_type": method_type,
            "eval_protocol": "Holdout",
            "auc": row.get("auc"),
            "f1": row.get("f1"),
            "recall": row.get("recall"),
            "fnr": row.get("fnr"),
            "dp_diff": row.get("dp_diff"),
            "fnr_gap": row.get("fnr_gap"),
            "worst_group_sensitivity": row.get("worst_group_sensitivity"),
            "macro_avg_fnr": row.get("macro_avg_fnr"),
        }, "holdout_raw"))

    return pd.DataFrame(rows)


def load_irm_results():
    """Loads ERM, IRM, and oracle target-domain generalisation results."""
    path = RESULTS_DIR / "irm" / "irm_experiment_results.csv"
    if not path.exists():
        print(f"  IRM results not found: {path}  ->  run irm_experiment.py first.")
        return None

    df = pd.read_csv(path)
    rows = []

    for _, row in df.iterrows():
        method = row.get("method")

        if method == "IRM source-invariant":
            method_type = "Invariant Learning"
        else:
            method_type = "Transfer/Oracle"

        rows.append(with_comparison_group({
            "model": method,
            "feature_set": "Without Sensitive Attributes + Proxy-Reduced Features",
            "method_type": method_type,
            "eval_protocol": "Source->Target Holdout",
            "auc": row.get("auc"),
            "f1": row.get("f1"),
            "recall": row.get("recall"),
            "fnr": row.get("fnr"),
            "dp_diff": row.get("dp_diff"),
            "fnr_gap": row.get("fnr_gap"),
            "worst_group_sensitivity": row.get("worst_group_sensitivity"),
            "macro_avg_fnr": row.get("macro_avg_fnr"),
        }, "transfer_target"))

    return pd.DataFrame(rows)


def load_vfae_results():
    """Loads VFAE test metrics."""
    path = RESULTS_DIR / "vfae" / "vfae_test_metrics.csv"
    if not path.exists():
        print(f"  VFAE results not found: {path}  ->  run vfae_experiment.py first.")
        return None

    row = pd.read_csv(path).iloc[0]
    return pd.DataFrame([with_comparison_group({
        "model": "VFAE",
        "feature_set": "Without Sensitive Attributes + Proxy-Reduced Features",
        "method_type": "Fair Deep Representation",
        "eval_protocol": "Holdout",
        "auc": row.get("auc"),
        "f1": row.get("f1"),
        "recall": row.get("recall"),
        "fnr": row.get("fnr"),
        "dp_diff": row.get("dp_diff"),
        "fnr_gap": row.get("fnr_gap"),
        "worst_group_sensitivity": row.get("worst_group_sensitivity"),
        "macro_avg_fnr": row.get("macro_avg_fnr"),
    }, "holdout_raw")])


def load_membership_inference_results():
    """Loads membership inference attack AUC results per epsilon."""
    path = RESULTS_DIR / "membership_inference" / "membership_inference_results.csv"
    if not path.exists():
        print(f"  Membership inference results not found: {path}  ->  run membership_inference_experiment.py first.")
        return None

    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        method = row.get("method", "")
        rows.append(with_comparison_group({
            "model": method,
            "feature_set": "Without Sensitive Attributes + Proxy-Reduced Features",
            "method_type": "Privacy Attack",
            "eval_protocol": "Holdout",
            "auc": row.get("attack_auc"),
            "f1": np.nan,
            "recall": np.nan,
            "fnr": np.nan,
            "dp_diff": np.nan,
            "fnr_gap": np.nan,
            "worst_group_sensitivity": np.nan,
            "macro_avg_fnr": np.nan,
            "attack_auc": row.get("attack_auc"),
            "attack_advantage": row.get("attack_advantage"),
        }, "privacy_audit"))
    return pd.DataFrame(rows)


def load_transfer_results():
    """Loads Oracle / Naive / IPS transfer experiment results."""
    path = RESULTS_DIR / "transfer" / "transfer_experiment_results.csv"
    if not path.exists():
        print(f"  Transfer results not found: {path}  -  run transfer_experiment.py first.")
        return None

    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        method = row["method"]
        method_type = "Transfer/Oracle" if "Oracle" in method else "Transfer Learning"
        rows.append(with_comparison_group({
            "model": method,
            "feature_set": "Without Sensitive Attributes",
            "method_type": method_type,
            "eval_protocol": "Holdout",
            "auc": row.get("auc"),
            "f1": row.get("f1"),
            "recall": row.get("recall"),
            "fnr": row.get("fnr"),
            "dp_diff": row.get("dp_diff"),
            "fnr_gap": row.get("fnr_gap"),
            "worst_group_sensitivity": row.get("worst_group_sensitivity"),
            "macro_avg_fnr": row.get("macro_avg_fnr"),
        }, "holdout_transfer"))
    return pd.DataFrame(rows)


def plot_scatter_comparison(df, output_dir):
    """Scatter plots: utility metric vs fairness metric, split by comparable protocol."""
    pairs = [
        ("auc", "dp_diff", "ROC-AUC", "DP Difference (Sex)"),
        ("auc", "fnr_gap", "ROC-AUC", "FNR Gap (Sex)"),
        ("recall", "worst_group_sensitivity", "Recall", "Worst-Group Sensitivity (Sex)"),
    ]

    for comparison_group, group_df in df.groupby("comparison_group"):
        if comparison_group == "privacy_audit":
            continue
        for x_col, y_col, x_label, y_label in pairs:
            plot_df = group_df.dropna(subset=[x_col, y_col])
            if plot_df.empty:
                continue

            plt.figure(figsize=(12, 7))

            for method_type, method_df in plot_df.groupby("method_type"):
                color = METHOD_COLORS.get(method_type, "gray")
                plt.scatter(method_df[x_col], method_df[y_col], s=90, label=method_type, color=color, zorder=3)
                for _, row in method_df.iterrows():
                    plt.annotate(
                        row["model"],
                        (row[x_col], row[y_col]),
                        fontsize=7,
                        xytext=(5, 5),
                        textcoords="offset points",
                        alpha=0.9,
                    )

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f"{comparison_group}: {x_label} vs {y_label}")
            plt.legend(title="Method type", fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            fname = f"scatter_{comparison_group}_{x_col}_vs_{y_col}.png"
            plt.savefig(output_dir / fname, dpi=300)
            plt.close()
            print(f"  Saved: {output_dir / fname}")


def plot_bar_comparison(df, output_dir):
    """Bar charts for key metrics within each directly comparable protocol."""
    metrics = [
        ("worst_group_sensitivity", "Worst-Group Sensitivity (Sex)  ↑ higher is better", False),
        ("macro_avg_fnr", "Macro-Averaged FNR (Sex)  ↓ lower is better", True),
        ("fnr_gap", "FNR Gap (Sex)  ↓ lower is fairer", True),
        ("recall", "Recall / Sensitivity  ↑ higher is better", False),
        ("f1", "F1-score", False),
        ("auc", "ROC-AUC - secondary ranking metric", False),
        ("dp_diff", "DP Difference (Sex)  ↓ lower is fairer", True),
    ]

    for comparison_group, group_df in df.groupby("comparison_group"):
        if comparison_group == "privacy_audit":
            continue
        for col, label, lower_better in metrics:
            plot_df = group_df.dropna(subset=[col]).copy()
            if plot_df.empty:
                continue

            plot_df = plot_df.sort_values(col, ascending=lower_better)
            colors = [METHOD_COLORS.get(t, "gray") for t in plot_df["method_type"]]

            fig, ax = plt.subplots(figsize=(max(10, len(plot_df) * 0.8), 5))
            bars = ax.bar(plot_df["model"], plot_df[col], color=colors)

            for bar, val in zip(bars, plot_df[col]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

            ax.set_xlabel("Model")
            ax.set_ylabel(label.split("  ")[0])
            ax.set_title(f"{comparison_group}: {label}")
            plt.xticks(rotation=40, ha="right", fontsize=8)
            plt.grid(axis="y", alpha=0.3)

            from matplotlib.patches import Patch
            legend_handles = [
                Patch(color=c, label=m) for m, c in METHOD_COLORS.items()
                if m in plot_df["method_type"].values
            ]
            ax.legend(handles=legend_handles, title="Method type", fontsize=8)

            plt.tight_layout()
            fname = f"bar_{comparison_group}_{col}.png"
            plt.savefig(output_dir / fname, dpi=300)
            plt.close()
            print(f"  Saved: {output_dir / fname}")


def main():
    output_dir = GRAPHS_DIR / "comparison"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Unified results comparison")
    print("=" * 80)

    print("\nLoading results...")
    cv_df = load_cv_results()
    dp_df = load_dp_results()
    dp_sgd_mlp_df = load_dp_sgd_mlp_results()
    vfae_df = load_vfae_results()
    irm_df = load_irm_results()
    transfer_df = load_transfer_results()
    mi_df = load_membership_inference_results()

    available = [
        df for df in [
            cv_df,
            dp_df,
            dp_sgd_mlp_df,
            vfae_df,
            irm_df,
            transfer_df,
            mi_df,
        ]
        if df is not None
    ]
    if not available:
        print("No results found. Run the experiment scripts first.")
        return

    unified_df = pd.concat(available, ignore_index=True)
    unified_df["primary_screening_score"] = np.nan
    unified_df["protocol_rank"] = np.nan

    comparable_mask = (
        unified_df["is_directly_comparable"]
        & unified_df["worst_group_sensitivity"].notna()
        & unified_df["macro_avg_fnr"].notna()
    )
    unified_df.loc[comparable_mask, "primary_screening_score"] = (
        unified_df.loc[comparable_mask, "worst_group_sensitivity"]
        - unified_df.loc[comparable_mask, "macro_avg_fnr"]
    )
    unified_df.loc[comparable_mask, "protocol_rank"] = (
        unified_df.loc[comparable_mask]
        .groupby("comparison_group")["primary_screening_score"]
        .rank(ascending=False, method="min")
    )

    output_csv = RESULTS_DIR / "unified_comparison.csv"
    unified_df.to_csv(output_csv, index=False)
    protocol_csv = RESULTS_DIR / "protocol_comparison_summary.csv"
    protocol_df = unified_df[unified_df["is_directly_comparable"]].copy()
    protocol_df.to_csv(protocol_csv, index=False)

    display_cols = [
        "model", "feature_set", "method_type", "eval_protocol",
        "comparison_group", "protocol_rank",
        "worst_group_sensitivity", "macro_avg_fnr", "fnr_gap",
        "recall", "fnr", "f1", "auc", "dp_diff",
    ]
    display_cols = [c for c in display_cols if c in unified_df.columns]

    print("\n" + "=" * 80)
    display_df = unified_df[display_cols].copy()
    display_df = display_df.replace({None: np.nan, "None": np.nan})
    print(display_df.round(4).to_string(index=False, na_rep=""))
    print("=" * 80)
    print(f"\nSaved: {output_csv}")
    print(f"Saved: {protocol_csv}")

    print("\nGenerating comparison plots...")
    plot_scatter_comparison(unified_df, output_dir)
    plot_bar_comparison(unified_df, output_dir)

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
