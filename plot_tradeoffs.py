import os
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR, GRAPHS_DIR


EXPERIMENTS = [
    "raw_dataset",
    "balanced_dataset",
]


def shorten_model_name(name):
    replacements = {
        "Logistic Regression": "LR",
        "Random Forest": "RF",
        "XGBoost": "XGB",
        "LightGBM": "LGBM",
        "CatBoost": "CatBoost",
        "MLP": "MLP",
        "Fairlearn-DP": "Fairlearn-DP",
    }

    return replacements.get(name, name)


def shorten_feature_set(name):
    replacements = {
        "Original Features": "Original",
        "Without Sensitive Attributes": "No-sensitive",
        "Without Sensitive Attributes + Proxy-Reduced Features": "Proxy-reduced",
    }

    return replacements.get(name, name)


def clean_filename(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace("-", "_")
        .replace("__", "_")
    )


def load_results(experiment_name):
    results_file = RESULTS_DIR / experiment_name / "final_cv_results_all_models.csv"

    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"Could not find {results_file}. Run main.py first."
        )

    df = pd.read_csv(results_file)

    required_cols = [
        "model",
        "feature_set",
        "accuracy_mean",
        "f1_mean",
        "auc_mean",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns in {results_file}: {missing_cols}"
        )

    df["model_short"] = df["model"].apply(shorten_model_name)
    df["feature_set_short"] = df["feature_set"].apply(shorten_feature_set)
    df["label"] = df["model_short"] + " / " + df["feature_set_short"]
    df["experiment"] = experiment_name

    return df


def plot_metric_vs_fairness(
    df,
    x_col,
    x_label,
    fairness_col,
    fairness_label,
    output_path,
):
    if x_col not in df.columns:
        print(f"Skipping plot: missing x column {x_col}")
        return

    if fairness_col not in df.columns:
        print(f"Skipping plot: missing fairness column {fairness_col}")
        return

    plt.figure(figsize=(13, 8))

    for feature_set in df["feature_set_short"].unique():
        subset = df[df["feature_set_short"] == feature_set]

        plt.scatter(
            subset[x_col],
            subset[fairness_col],
            s=70,
            label=feature_set,
        )

        for _, row in subset.iterrows():
            plt.annotate(
                row["model_short"],
                (row[x_col], row[fairness_col]),
                fontsize=8,
                alpha=0.85,
                xytext=(5, 5),
                textcoords="offset points",
            )

    plt.xlabel(x_label)
    plt.ylabel(fairness_label)
    plt.title(f"Diabetes Risk Prediction: {x_label} vs {fairness_label}")
    plt.legend(title="Feature set")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_feature_set_comparison(df, metric_col, metric_label, output_path):
    if metric_col not in df.columns:
        print(f"Skipping feature-set comparison: missing column {metric_col}")
        return

    pivot_df = df.pivot(
        index="model_short",
        columns="feature_set_short",
        values=metric_col,
    )

    ordered_cols = [
        col for col in ["Original", "No-sensitive", "Proxy-reduced"]
        if col in pivot_df.columns
    ]

    if not ordered_cols:
        print(f"Skipping feature-set comparison for {metric_col}: no feature sets found")
        return

    pivot_df = pivot_df[ordered_cols]

    ax = pivot_df.plot(kind="bar", figsize=(12, 7))

    ax.set_xlabel("Model")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Diabetes Risk Prediction: {metric_label} by Feature Set")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_clinical_ranking(df, output_path):
    """
    Ranks models using clinically motivated fairness metrics.

    Higher worst-group sensitivity is better.
    Lower macro-averaged FNR is better.
    Lower FNR gap is better.
    """
    needed_cols = [
        "model",
        "feature_set",
        "auc_mean",
        "sex_group_worst_group_sensitivity_mean",
        "sex_group_macro_avg_fnr_mean",
        "sex_group_fnr_gap_mean",
    ]

    missing_cols = [col for col in needed_cols if col not in df.columns]

    if missing_cols:
        print(f"Skipping clinical ranking plot. Missing columns: {missing_cols}")
        return

    ranking_df = df.copy()

    ranking_df["clinical_score"] = (
        ranking_df["sex_group_worst_group_sensitivity_mean"]
        - ranking_df["sex_group_macro_avg_fnr_mean"]
        - ranking_df["sex_group_fnr_gap_mean"]
    )

    ranking_df = ranking_df.sort_values("clinical_score", ascending=False)

    ranking_df["label"] = (
        ranking_df["model_short"] + " / " + ranking_df["feature_set_short"]
    )

    top_df = ranking_df.head(15).iloc[::-1]

    plt.figure(figsize=(12, 8))
    plt.barh(top_df["label"], top_df["clinical_score"])

    plt.xlabel("Clinical fairness score")
    plt.ylabel("Model / Feature set")
    plt.title(
        "Clinical Fairness Ranking\n"
        "Higher worst-group sensitivity, lower macro FNR, lower FNR gap"
    )
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def run_plots_for_experiment(experiment_name):
    print("\n" + "=" * 80)
    print(f"Creating tradeoff plots for: {experiment_name}")
    print("=" * 80)

    df = load_results(experiment_name)

    experiment_graph_dir = GRAPHS_DIR / experiment_name
    os.makedirs(experiment_graph_dir, exist_ok=True)

    fairness_plots = [
        (
            "sex_group_dp_diff_mean",
            "Sex Demographic Parity Difference",
        ),
        (
            "age_group_dp_diff_mean",
            "Age Demographic Parity Difference",
        ),
        (
            "income_group_dp_diff_mean",
            "Income Demographic Parity Difference",
        ),
        (
            "education_group_dp_diff_mean",
            "Education Demographic Parity Difference",
        ),
        (
            "intersection_group_equalized_odds_diff_mean",
            "Intersection Equalized Odds Difference",
        ),
        (
            "sex_group_fnr_gap_mean",
            "Sex FNR Gap",
        ),
        (
            "age_group_fnr_gap_mean",
            "Age FNR Gap",
        ),
        (
            "intersection_group_fnr_gap_mean",
            "Intersection FNR Gap",
        ),
        (
            "sex_group_macro_avg_fnr_mean",
            "Sex Macro-Averaged FNR",
        ),
        (
            "age_group_macro_avg_fnr_mean",
            "Age Macro-Averaged FNR",
        ),
        (
            "intersection_group_macro_avg_fnr_mean",
            "Intersection Macro-Averaged FNR",
        ),
        (
            "sex_group_worst_group_sensitivity_mean",
            "Sex Worst-Group Sensitivity",
        ),
        (
            "age_group_worst_group_sensitivity_mean",
            "Age Worst-Group Sensitivity",
        ),
        (
            "intersection_group_worst_group_sensitivity_mean",
            "Intersection Worst-Group Sensitivity",
        ),
    ]

    x_metrics = [
        ("accuracy_mean", "Accuracy"),
        ("f1_mean", "F1-score"),
        ("auc_mean", "ROC-AUC"),
        ("recall_mean", "Recall / Sensitivity"),
        ("fnr_mean", "False Negative Rate"),
        ("brier_mean", "Brier Score"),
    ]

    for fairness_col, fairness_label in fairness_plots:
        safe_fairness_name = clean_filename(fairness_label)

        for x_col, x_label in x_metrics:
            safe_x_name = clean_filename(x_label)

            output_path = experiment_graph_dir / f"{safe_x_name}_vs_{safe_fairness_name}.png"

            plot_metric_vs_fairness(
                df=df,
                x_col=x_col,
                x_label=x_label,
                fairness_col=fairness_col,
                fairness_label=fairness_label,
                output_path=output_path,
            )

    feature_comparison_metrics = [
        ("accuracy_mean", "Accuracy"),
        ("f1_mean", "F1-score"),
        ("auc_mean", "ROC-AUC"),
        ("recall_mean", "Recall / Sensitivity"),
        ("fnr_mean", "False Negative Rate"),
        ("brier_mean", "Brier Score"),
        ("sex_group_dp_diff_mean", "Sex Demographic Parity Difference"),
        ("sex_group_fnr_gap_mean", "Sex FNR Gap"),
        ("sex_group_macro_avg_fnr_mean", "Sex Macro-Averaged FNR"),
        ("sex_group_worst_group_sensitivity_mean", "Sex Worst-Group Sensitivity"),
        ("intersection_group_equalized_odds_diff_mean", "Intersection Equalized Odds Difference"),
        ("intersection_group_fnr_gap_mean", "Intersection FNR Gap"),
        ("intersection_group_macro_avg_fnr_mean", "Intersection Macro-Averaged FNR"),
        ("intersection_group_worst_group_sensitivity_mean", "Intersection Worst-Group Sensitivity"),
    ]

    for metric_col, metric_label in feature_comparison_metrics:
        safe_metric_name = clean_filename(metric_label)
        output_path = experiment_graph_dir / f"feature_set_{safe_metric_name}_comparison.png"

        plot_feature_set_comparison(
            df=df,
            metric_col=metric_col,
            metric_label=metric_label,
            output_path=output_path,
        )

    clinical_output_path = experiment_graph_dir / "clinical_fairness_ranking.png"
    plot_clinical_ranking(df, clinical_output_path)

    print(f"\nFinished plots for: {experiment_name}")


def main():
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    for experiment_name in EXPERIMENTS:
        try:
            run_plots_for_experiment(experiment_name)
        except FileNotFoundError as error:
            print(f"\nSkipping {experiment_name}: {error}")

    print("\nTradeoff plots complete.")


if __name__ == "__main__":
    main()