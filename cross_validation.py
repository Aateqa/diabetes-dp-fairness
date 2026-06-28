import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from config import RESULTS_DIR, RANDOM_STATE, N_SPLITS
from models import get_models
from metrics import append_metric_dict


def cross_validate_models(feature_sets, y, fairness_df):
    """
    Runs stratified cross-validation for all models across all feature sets.

    For each fold:
    - train the model
    - predict labels
    - predict probabilities where available
    - compute utility and fairness metrics
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for feature_set_name, X in feature_sets.items():
        print("\n" + "=" * 80)
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

                fairness_test = fairness_df.iloc[test_idx].copy()

                fold_model = clone(model)

                fold_model.fit(X_train, y_train)

                y_pred = fold_model.predict(X_test)

                if hasattr(fold_model, "predict_proba"):
                    y_prob = fold_model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = y_pred

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

            mean_row = {
                "model": model_name,
                "feature_set": feature_set_name,
            }

            std_row = {
                "model": model_name,
                "feature_set": feature_set_name,
            }

            numeric_cols = fold_df.select_dtypes(include="number").columns

            for col in numeric_cols:
                mean_row[f"{col}_mean"] = fold_df[col].mean()
                std_row[f"{col}_std"] = fold_df[col].std()

            combined_row = {
                "model": model_name,
                "feature_set": feature_set_name,
            }

            for col in numeric_cols:
                combined_row[f"{col}_mean"] = fold_df[col].mean()
                combined_row[f"{col}_std"] = fold_df[col].std()

            all_results.append(combined_row)

    results_df = pd.DataFrame(all_results)

    output_path = f"{RESULTS_DIR}/final_cv_results_all_models.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("Cross-validation complete")
    print(f"Saved: {output_path}")
    print("=" * 80)

    return results_df


def format_results_for_display(results_df):
    """
    Creates a readable version of the CV results.
    """
    display_cols = [
        "model",
        "feature_set",
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "auc_mean",
        "sex_group_dp_diff_mean",
        "age_group_dp_diff_mean",
        "income_group_dp_diff_mean",
        "education_group_dp_diff_mean",
        "intersection_group_equalized_odds_diff_mean",
    ]

    existing_cols = [col for col in display_cols if col in results_df.columns]

    display_df = results_df[existing_cols].copy()

    numeric_cols = display_df.select_dtypes(include="number").columns
    display_df[numeric_cols] = display_df[numeric_cols].round(4)

    return display_df


if __name__ == "__main__":
    from data_loader_diabetes import load_diabetes_data

    feature_sets, y, fairness_df, df = load_diabetes_data(print_summary=False)

    results_df = cross_validate_models(
        feature_sets=feature_sets,
        y=y,
        fairness_df=fairness_df,
    )

    print("\nReadable results:")
    print(format_results_for_display(results_df))