import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import (
    RESULTS_DIR,
    GRAPHS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)

from data_loader_diabetes import load_raw_diabetes_data
from metrics import safe_auc, get_probabilities, counterfactual_fairness_ratio
from models.tree_ensemble import make_xgb_model


N_COUNTERFACTUAL_SAMPLES = 500


def flip_sex_counterfactual(X_original):
    """
    Flips Sex from female to male and male to female.

    BRFSS Sex encoding:
    0 = female
    1 = male

    If Sex is not present in the feature set, this returns None.
    """
    if "Sex" not in X_original.columns:
        return None

    X_flipped = X_original.copy()
    X_flipped["Sex"] = 1 - X_flipped["Sex"]

    return X_flipped


def shift_age_counterfactual(X_original):
    """
    Shifts age group from young to middle-aged.

    BRFSS Age is coded 1 to 13.

    Approximate rule:
    - Young: Age 1-5
    - Middle-aged: Age 6-9
    - Older: Age 10-13

    For the counterfactual:
    - If Age <= 5, set Age to 7.
    - If Age is already middle-aged or older, keep it unchanged.

    If engineered age_bmi_risk is present, update it too so it stays consistent.
    """
    if "Age" not in X_original.columns:
        return None

    X_shifted = X_original.copy()

    young_mask = X_shifted["Age"] <= 5
    X_shifted.loc[young_mask, "Age"] = 7

    if "age_bmi_risk" in X_shifted.columns:
        X_shifted["age_bmi_risk"] = (
            X_shifted["Age"] * (X_shifted["BMI"] >= 30).astype("int8")
        ).astype("int16")

    return X_shifted


def compute_counterfactual_change_summary(
    name,
    y_true,
    original_pred,
    original_prob,
    counterfactual_pred,
    counterfactual_prob,
):
    """
    Computes how much predictions and probabilities changed after counterfactual editing.

    counterfactual_fairness_ratio is the fraction of individuals whose prediction
    changes when the sensitive attribute is intervened upon (Kusner et al., 2017).
    A model is perfectly counterfactually fair when this ratio is 0.
    """
    prediction_changed = original_pred != counterfactual_pred
    probability_change = counterfactual_prob - original_prob

    return {
        "counterfactual": name,
        "n_samples": len(original_pred),
        "counterfactual_fairness_ratio": float(prediction_changed.mean()),
        "n_predictions_changed": int(prediction_changed.sum()),
        "percent_predictions_changed": float(prediction_changed.mean() * 100),
        "mean_absolute_probability_change": float(np.mean(np.abs(probability_change))),
        "mean_signed_probability_change": float(np.mean(probability_change)),
        "max_absolute_probability_change": float(np.max(np.abs(probability_change))),
        "original_accuracy": accuracy_score(y_true, original_pred),
        "counterfactual_accuracy": accuracy_score(y_true, counterfactual_pred),
        "original_auc": safe_auc(y_true, original_prob),
        "counterfactual_auc": safe_auc(y_true, counterfactual_prob),
    }


def save_sample_level_changes(
    output_path,
    sample_ids,
    y_true,
    original_pred,
    original_prob,
    counterfactual_pred,
    counterfactual_prob,
):
    sample_df = pd.DataFrame({
        "sample_id": sample_ids,
        "y_true": y_true,
        "original_pred": original_pred,
        "counterfactual_pred": counterfactual_pred,
        "prediction_changed": original_pred != counterfactual_pred,
        "original_probability": original_prob,
        "counterfactual_probability": counterfactual_prob,
        "probability_change": counterfactual_prob - original_prob,
        "absolute_probability_change": np.abs(counterfactual_prob - original_prob),
    })

    sample_df.to_csv(output_path, index=False)
    print(f"Saved sample-level changes: {output_path}")


def plot_counterfactual_summary(summary_df, output_path):
    plt.figure(figsize=(9, 6))

    plt.bar(
        summary_df["counterfactual"],
        summary_df["percent_predictions_changed"],
    )

    plt.ylabel("% predictions changed")
    plt.xlabel("Counterfactual intervention")
    plt.title("Counterfactual Sensitivity of Diabetes Predictions")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_probability_change_summary(summary_df, output_path):
    plt.figure(figsize=(9, 6))

    plt.bar(
        summary_df["counterfactual"],
        summary_df["mean_absolute_probability_change"],
    )

    plt.ylabel("Mean absolute probability change")
    plt.xlabel("Counterfactual intervention")
    plt.title("Counterfactual Probability Shift")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot: {output_path}")


def run_counterfactual_experiment():
    os.makedirs(RESULTS_DIR / "counterfactual", exist_ok=True)
    os.makedirs(GRAPHS_DIR / "counterfactual", exist_ok=True)

    print("\n" + "=" * 80)
    print("Running counterfactual diabetes experiment")
    print("=" * 80)

    feature_sets, y, fairness_df, df = load_raw_diabetes_data(print_summary=False)

    # For Sex and Age flipping, we need a feature set that actually contains Sex and Age.
    # So we use Original Features here.
    feature_set_name = "Original Features"
    X = feature_sets[feature_set_name]

    print(f"Feature set: {feature_set_name}")
    print(f"X shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_xgb_model(random_state=RANDOM_STATE)

    print("\nTraining XGBoost model for counterfactual analysis...")
    model.fit(X_train, y_train)

    sample_size = min(N_COUNTERFACTUAL_SAMPLES, len(X_test))

    X_sample = X_test.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    y_sample = y_test.loc[X_sample.index]

    sample_ids = X_sample.index.to_list()

    original_pred = model.predict(X_sample)
    original_prob = get_probabilities(model, X_sample)

    summaries = []

    # 1. Sex flip: female -> male, male -> female.
    X_sex_flipped = flip_sex_counterfactual(X_sample)

    if X_sex_flipped is not None:
        sex_pred = model.predict(X_sex_flipped)
        sex_prob = get_probabilities(model, X_sex_flipped)

        sex_summary = compute_counterfactual_change_summary(
            name="Sex flip",
            y_true=y_sample,
            original_pred=original_pred,
            original_prob=original_prob,
            counterfactual_pred=sex_pred,
            counterfactual_prob=sex_prob,
        )

        summaries.append(sex_summary)

        save_sample_level_changes(
            output_path=RESULTS_DIR / "counterfactual" / "sex_flip_sample_changes.csv",
            sample_ids=sample_ids,
            y_true=y_sample.to_numpy(),
            original_pred=original_pred,
            original_prob=original_prob,
            counterfactual_pred=sex_pred,
            counterfactual_prob=sex_prob,
        )
    else:
        print("Skipping Sex flip: Sex column not found in feature set.")

    # 2. Age group shift: young -> middle-aged.
    X_age_shifted = shift_age_counterfactual(X_sample)

    if X_age_shifted is not None:
        age_pred = model.predict(X_age_shifted)
        age_prob = get_probabilities(model, X_age_shifted)

        age_summary = compute_counterfactual_change_summary(
            name="Age young→middle",
            y_true=y_sample,
            original_pred=original_pred,
            original_prob=original_prob,
            counterfactual_pred=age_pred,
            counterfactual_prob=age_prob,
        )

        summaries.append(age_summary)

        save_sample_level_changes(
            output_path=RESULTS_DIR / "counterfactual" / "age_shift_sample_changes.csv",
            sample_ids=sample_ids,
            y_true=y_sample.to_numpy(),
            original_pred=original_pred,
            original_prob=original_prob,
            counterfactual_pred=age_pred,
            counterfactual_prob=age_prob,
        )
    else:
        print("Skipping Age shift: Age column not found in feature set.")

    summary_df = pd.DataFrame(summaries)

    summary_output = RESULTS_DIR / "counterfactual" / "counterfactual_summary.csv"
    plot_output = GRAPHS_DIR / "counterfactual" / "counterfactual_prediction_changes.png"
    prob_plot_output = GRAPHS_DIR / "counterfactual" / "counterfactual_probability_changes.png"

    summary_df.to_csv(summary_output, index=False)

    print(f"\nSaved summary: {summary_output}")
    print("\nCounterfactual summary:")
    print(summary_df.round(4))

    plot_counterfactual_summary(
        summary_df=summary_df,
        output_path=plot_output,
    )

    plot_probability_change_summary(
        summary_df=summary_df,
        output_path=prob_plot_output,
    )

    print("\nCounterfactual experiment complete.")

    return summary_df


if __name__ == "__main__":
    run_counterfactual_experiment()