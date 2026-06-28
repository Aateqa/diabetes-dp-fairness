import os
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR, GRAPHS_DIR


RESULTS_FILE = f"{RESULTS_DIR}/final_cv_results_all_models.csv"


def shorten_model_name(name):
    replacements = {
        "Logistic Regression": "LR",
        "Decision Tree": "DT",
        "Random Forest": "RF",
        "XGBoost": "XGB",
        "LightGBM": "LGBM",
        "CatBoost": "CatBoost",
        "Stacking Ensemble": "Stacking",
    }

    return replacements.get(name, name)


def shorten_feature_set(name):
    replacements = {
        "Original Features": "Original",
        "Without Sensitive Attributes": "No-sensitive",
        "Without Sensitive Attributes + Proxy-Reduced Features": "Proxy-reduced",
    }

    return replacements.get(name, name)


def load_results():
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(
            f"Could not find {RESULTS_FILE}. Run main_diabetes.py or cross_validation.py first."
        )

    df = pd.read_csv(RESULTS_FILE)

    required_cols = [
        "model",
        "feature_set",
        "accuracy_mean",
        "f1_mean",
        "auc_mean",
        "sex_group_dp_diff_mean",
        "age_group_dp_diff_mean",
        "income_group_dp_diff_mean",
        "education_group_dp_diff_mean",
        "intersection_group_equalized_odds_diff_mean",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in results CSV: {missing_cols}")

    df["model_short"] = df["model"].apply(shorten_model_name)
    df["feature_set_short"] = df["feature_set"].apply(shorten_feature_set)
    df["label"] = df["model_short"] + " / " + df["feature_set_short"]

    return df


def plot_accuracy_vs_fairness(df, fairness_col, fairness_label, output_name):
    plt.figure(figsize=(13, 8))

    for feature_set in df["feature_set_short"].unique():
        subset = df[df["feature_set_short"] == feature_set]

        plt.scatter(
            subset["accuracy_mean"],
            subset[fairness_col],
            s=70,
            label=feature_set,
        )

        for _, row in subset.iterrows():
            plt.annotate(
                row["model_short"],
                (row["accuracy_mean"], row[fairness_col]),
                fontsize=8,
                alpha=0.85,
                xytext=(5, 5),
                textcoords="offset points",
            )

    plt.xlabel("Accuracy")
    plt.ylabel(fairness_label)
    plt.title(f"Diabetes Risk Prediction: Accuracy vs {fairness_label}")
    plt.legend(title="Feature set")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = f"{GRAPHS_DIR}/{output_name}"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_f1_vs_fairness(df, fairness_col, fairness_label, output_name):
    plt.figure(figsize=(13, 8))

    for feature_set in df["feature_set_short"].unique():
        subset = df[df["feature_set_short"] == feature_set]

        plt.scatter(
            subset["f1_mean"],
            subset[fairness_col],
            s=70,
            label=feature_set,
        )

        for _, row in subset.iterrows():
            plt.annotate(
                row["model_short"],
                (row["f1_mean"], row[fairness_col]),
                fontsize=8,
                alpha=0.85,
                xytext=(5, 5),
                textcoords="offset points",
            )

    plt.xlabel("F1-score")
    plt.ylabel(fairness_label)
    plt.title(f"Diabetes Risk Prediction: F1-score vs {fairness_label}")
    plt.legend(title="Feature set")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = f"{GRAPHS_DIR}/{output_name}"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_auc_vs_fairness(df, fairness_col, fairness_label, output_name):
    plt.figure(figsize=(13, 8))

    for feature_set in df["feature_set_short"].unique():
        subset = df[df["feature_set_short"] == feature_set]

        plt.scatter(
            subset["auc_mean"],
            subset[fairness_col],
            s=70,
            label=feature_set,
        )

        for _, row in subset.iterrows():
            plt.annotate(
                row["model_short"],
                (row["auc_mean"], row[fairness_col]),
                fontsize=8,
                alpha=0.85,
                xytext=(5, 5),
                textcoords="offset points",
            )

    plt.xlabel("ROC-AUC")
    plt.ylabel(fairness_label)
    plt.title(f"Diabetes Risk Prediction: ROC-AUC vs {fairness_label}")
    plt.legend(title="Feature set")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = f"{GRAPHS_DIR}/{output_name}"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_feature_set_comparison(df, metric_col, metric_label, output_name):
    pivot_df = df.pivot(
        index="model_short",
        columns="feature_set_short",
        values=metric_col,
    )

    pivot_df = pivot_df[
        [col for col in ["Original", "No-sensitive", "Proxy-reduced"] if col in pivot_df.columns]
    ]

    plt.figure(figsize=(12, 7))
    pivot_df.plot(kind="bar", figsize=(12, 7))

    plt.xlabel("Model")
    plt.ylabel(metric_label)
    plt.title(f"Diabetes Risk Prediction: {metric_label} by Feature Set")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = f"{GRAPHS_DIR}/{output_name}"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    df = load_results()

    fairness_plots = [
        (
            "sex_group_dp_diff_mean",
            "Sex Demographic Parity Difference",
            "accuracy_vs_sex_dp_diff.png",
            "f1_vs_sex_dp_diff.png",
            "auc_vs_sex_dp_diff.png",
        ),
        (
            "age_group_dp_diff_mean",
            "Age Demographic Parity Difference",
            "accuracy_vs_age_dp_diff.png",
            "f1_vs_age_dp_diff.png",
            "auc_vs_age_dp_diff.png",
        ),
        (
            "income_group_dp_diff_mean",
            "Income Demographic Parity Difference",
            "accuracy_vs_income_dp_diff.png",
            "f1_vs_income_dp_diff.png",
            "auc_vs_income_dp_diff.png",
        ),
        (
            "education_group_dp_diff_mean",
            "Education Demographic Parity Difference",
            "accuracy_vs_education_dp_diff.png",
            "f1_vs_education_dp_diff.png",
            "auc_vs_education_dp_diff.png",
        ),
        (
            "intersection_group_equalized_odds_diff_mean",
            "Intersection Equalized Odds Difference",
            "accuracy_vs_intersection_equalized_odds.png",
            "f1_vs_intersection_equalized_odds.png",
            "auc_vs_intersection_equalized_odds.png",
        ),
    ]

    for fairness_col, fairness_label, accuracy_file, f1_file, auc_file in fairness_plots:
        plot_accuracy_vs_fairness(
            df=df,
            fairness_col=fairness_col,
            fairness_label=fairness_label,
            output_name=accuracy_file,
        )

        plot_f1_vs_fairness(
            df=df,
            fairness_col=fairness_col,
            fairness_label=fairness_label,
            output_name=f1_file,
        )

        plot_auc_vs_fairness(
            df=df,
            fairness_col=fairness_col,
            fairness_label=fairness_label,
            output_name=auc_file,
        )

    plot_feature_set_comparison(
        df=df,
        metric_col="accuracy_mean",
        metric_label="Accuracy",
        output_name="feature_set_accuracy_comparison.png",
    )

    plot_feature_set_comparison(
        df=df,
        metric_col="f1_mean",
        metric_label="F1-score",
        output_name="feature_set_f1_comparison.png",
    )

    plot_feature_set_comparison(
        df=df,
        metric_col="auc_mean",
        metric_label="ROC-AUC",
        output_name="feature_set_auc_comparison.png",
    )

    plot_feature_set_comparison(
        df=df,
        metric_col="intersection_group_equalized_odds_diff_mean",
        metric_label="Intersection Equalized Odds Difference",
        output_name="feature_set_intersection_equalized_odds_comparison.png",
    )

    print("\nTradeoff plots complete.")


if __name__ == "__main__":
    main()