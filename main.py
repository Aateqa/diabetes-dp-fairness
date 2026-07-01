import random

import numpy as np
import pandas as pd

from data_loader_diabetes import (
    load_raw_diabetes_data,
    load_balanced_diabetes_data,
)
from cross_validation import cross_validate_models, format_results_for_display
from config import RESULTS_DIR, RANDOM_STATE


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)


def run_experiment(loader_fn, experiment_name):
    print("\n" + "=" * 80)
    print(f"Starting experiment: {experiment_name}")
    print("=" * 80)

    feature_sets, y, fairness_df, df = loader_fn(print_summary=True)

    results_df = cross_validate_models(
        feature_sets=feature_sets,
        y=y,
        fairness_df=fairness_df,
        experiment_name=experiment_name,
    )

    readable_results = format_results_for_display(results_df)

    output_dir = RESULTS_DIR / experiment_name
    readable_path = output_dir / "final_cv_results_readable.csv"

    readable_results.to_csv(readable_path, index=False)

    print("\nReadable results:")
    print(readable_results)

    print("\nSaved:")
    print(output_dir / "final_cv_results_all_models.csv")
    print(readable_path)

    return results_df


def main():
    set_seed()
    print("\nStarting diabetes fairness analysis...")

    raw_results = run_experiment(
        loader_fn=load_raw_diabetes_data,
        experiment_name="raw_dataset",
    )

    balanced_results = run_experiment(
        loader_fn=load_balanced_diabetes_data,
        experiment_name="balanced_dataset",
    )

    combined = pd.concat([raw_results, balanced_results], ignore_index=True)

    combined_path = RESULTS_DIR / "combined_raw_and_balanced_cv_results.csv"
    combined.to_csv(combined_path, index=False)

    print("\nAll experiments complete.")
    print(f"Combined results saved to: {combined_path}")


if __name__ == "__main__":
    main()