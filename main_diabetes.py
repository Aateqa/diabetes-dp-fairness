from data_loader_diabetes import load_diabetes_data
from cross_validation import cross_validate_models, format_results_for_display


def main():
    print("\nStarting diabetes fairness analysis...")

    feature_sets, y, fairness_df, df = load_diabetes_data(print_summary=True)

    results_df = cross_validate_models(
        feature_sets=feature_sets,
        y=y,
        fairness_df=fairness_df,
    )

    readable_results = format_results_for_display(results_df)

    print("\nFinal readable results:")
    print(readable_results)

    readable_results.to_csv(
        "results/final_cv_results_readable.csv",
        index=False,
    )

    print("\nSaved:")
    print("results/final_cv_results_all_models.csv")
    print("results/final_cv_results_readable.csv")

    print("\nDiabetes fairness baseline complete.")


if __name__ == "__main__":
    main()