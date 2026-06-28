import os
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import RESULTS_DIR, GRAPHS_DIR, RANDOM_STATE, TEST_SIZE
from data_loader_diabetes import load_diabetes_data


def make_xgb_model():
    """
    XGBoost model used for SHAP explainability.
    """
    return XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def clean_name(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("+", "")
        .replace("-", "_")
        .replace("__", "_")
    )


def run_shap_for_feature_set(feature_set_name, X, y):
    """
    Trains XGBoost on one feature set and creates SHAP importance outputs.
    """

    print("\n" + "=" * 80)
    print(f"Running SHAP for: {feature_set_name}")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_xgb_model()
    model.fit(X_train, y_train)

    # Use a sample for speed because the dataset is large.
    shap_sample_size = min(5000, len(X_test))
    X_shap = X_test.sample(
        n=shap_sample_size,
        random_state=RANDOM_STATE,
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    feature_importance = pd.DataFrame({
        "feature": X_shap.columns,
        "mean_abs_shap": abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    safe_name = clean_name(feature_set_name)

    csv_output_path = f"{RESULTS_DIR}/xgb_{safe_name}_shap_importance.csv"
    graph_output_path = f"{GRAPHS_DIR}/xgb_{safe_name}_shap_importance.png"

    feature_importance.to_csv(csv_output_path, index=False)

    plt.figure(figsize=(10, 7))
    top_features = feature_importance.head(15).iloc[::-1]

    plt.barh(
        top_features["feature"],
        top_features["mean_abs_shap"],
    )

    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title(f"Diabetes Risk Prediction: SHAP Importance\n{feature_set_name}")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(graph_output_path, dpi=300)
    plt.close()

    print(f"Saved: {csv_output_path}")
    print(f"Saved: {graph_output_path}")

    return feature_importance


def plot_shap_feature_comparison(all_importances):
    """
    Creates a comparison plot showing top SHAP features across feature sets.
    """

    comparison_rows = []

    for feature_set_name, importance_df in all_importances.items():
        top_df = importance_df.head(10).copy()
        top_df["feature_set"] = feature_set_name
        comparison_rows.append(top_df)

    combined_df = pd.concat(comparison_rows, ignore_index=True)

    output_csv = f"{RESULTS_DIR}/shap_feature_comparison.csv"
    combined_df.to_csv(output_csv, index=False)

    plt.figure(figsize=(12, 8))

    for feature_set_name in combined_df["feature_set"].unique():
        subset = combined_df[combined_df["feature_set"] == feature_set_name]

        plt.scatter(
            subset["mean_abs_shap"],
            subset["feature"],
            s=70,
            label=feature_set_name,
        )

    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title("Diabetes Risk Prediction: SHAP Feature Importance Comparison")
    plt.legend(fontsize=8)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    output_graph = f"{GRAPHS_DIR}/shap_feature_comparison.png"
    plt.savefig(output_graph, dpi=300)
    plt.close()

    print(f"Saved: {output_csv}")
    print(f"Saved: {output_graph}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    feature_sets, y, fairness_df, df = load_diabetes_data(print_summary=False)

    all_importances = {}

    for feature_set_name, X in feature_sets.items():
        importance_df = run_shap_for_feature_set(
            feature_set_name=feature_set_name,
            X=X,
            y=y,
        )

        all_importances[feature_set_name] = importance_df

    plot_shap_feature_comparison(all_importances)

    print("\nSHAP analysis complete.")


if __name__ == "__main__":
    main()