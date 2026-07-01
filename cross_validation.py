import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import copy

from config import RESULTS_DIR, RANDOM_STATE, N_SPLITS, FAIRLEARN_SENSITIVE_ATTRIBUTE
from models import get_models
from metrics import append_metric_dict, get_probabilities


def fit_model(model_name, model, X_train, y_train, fairness_train):
    """
    Handles normal sklearn models, MLP sample weights, and Fairlearn's special API.
    """
    if model_name in ("Fairlearn-DP", "VFAE"):
        sensitive_features = fairness_train[FAIRLEARN_SENSITIVE_ATTRIBUTE]
        model.fit(X_train, y_train, sensitive_features=sensitive_features)
        return model

    if model_name == "MLP":
        model.fit(X_train, y_train)
        return model

    model.fit(X_train, y_train)
    return model



def cross_validate_models(feature_sets, y, fairness_df, experiment_name="raw_dataset"):
    output_dir = RESULTS_DIR / experiment_name
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for feature_set_name, X in feature_sets.items():
        print("\n" + "=" * 80)
        print(f"Experiment: {experiment_name}")
        print(f"Running feature set: {feature_set_name}")
        print("=" * 80)

        models = get_models(random_state=RANDOM_STATE)

        for model_name, model in models.items():
            print(f"\nModel: {model_name}")

            fold_results = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
                print(f"  Fold {fold_idx}/{N_SPLITS}")

                X_train = X.iloc[train_idx].copy()
                X_test = X.iloc[test_idx].copy()

                y_train = y.iloc[train_idx].copy()
                y_test = y.iloc[test_idx].copy()

                fairness_train = fairness_df.iloc[train_idx].copy()
                fairness_test = fairness_df.iloc[test_idx].copy()

                try:
                    fold_model = clone(model)
                except RuntimeError:
                    fold_model = copy.deepcopy(model)

                fold_model = fit_model(
                    model_name=model_name,
                    model=fold_model,
                    X_train=X_train,
                    y_train=y_train,
                    fairness_train=fairness_train,
                )

                y_pred = fold_model.predict(X_test)
                y_prob = get_probabilities(fold_model, X_test)

                append_metric_dict(
                    results_list=fold_results,
                    model_name=model_name,
                    feature_set_name=feature_set_name,
                    y_true=y_test,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    fairness_df=fairness_test,
                )

            fold_df = pd.DataFrame(fold_results)

            safe_name = (
                f"{model_name}_{feature_set_name}"
                .replace(" ", "_")
                .replace("+", "plus")
            )
            fold_output_path = output_dir / f"{safe_name}_fold_results.csv"
            fold_df.to_csv(fold_output_path, index=False)

            combined_row = {
                "model": model_name,
                "feature_set": feature_set_name,
                "experiment": experiment_name,
            }

            numeric_cols = fold_df.select_dtypes(include="number").columns

            for col in numeric_cols:
                combined_row[f"{col}_mean"] = fold_df[col].mean()
                combined_row[f"{col}_std"] = fold_df[col].std()

            all_results.append(combined_row)

    results_df = pd.DataFrame(all_results)

    output_path = output_dir / "final_cv_results_all_models.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("Cross-validation complete")
    print(f"Saved: {output_path}")
    print("=" * 80)

    return results_df


def format_results_for_display(results_df):
    display_cols = [
        "experiment",
        "model",
        "feature_set",
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "fnr_mean",
        "f1_mean",
        "auc_mean",
        "brier_mean",

        "sex_group_dp_diff_mean",
        "sex_group_fnr_gap_mean",
        "sex_group_worst_group_sensitivity_mean",
        "sex_group_macro_avg_fnr_mean",
        "sex_group_group_auc_mean_mean",
        "sex_group_group_brier_mean_mean",

        "age_group_dp_diff_mean",
        "age_group_fnr_gap_mean",
        "age_group_worst_group_sensitivity_mean",
        "age_group_macro_avg_fnr_mean",

        "income_group_dp_diff_mean",
        "education_group_dp_diff_mean",
        "intersection_group_equalized_odds_diff_mean",
        "intersection_group_fnr_gap_mean",
        "intersection_group_worst_group_sensitivity_mean",
        "intersection_group_macro_avg_fnr_mean",
    ]

    existing_cols = [col for col in display_cols if col in results_df.columns]

    display_df = results_df[existing_cols].copy()

    numeric_cols = display_df.select_dtypes(include="number").columns
    display_df[numeric_cols] = display_df[numeric_cols].round(4)

    return display_df


if __name__ == "__main__":
    from data_loader_diabetes import load_raw_diabetes_data

    feature_sets, y, fairness_df, df = load_raw_diabetes_data(print_summary=False)

    results_df = cross_validate_models(
        feature_sets=feature_sets,
        y=y,
        fairness_df=fairness_df,
        experiment_name="raw_dataset",
    )

    print("\nReadable results:")
    print(format_results_for_display(results_df))