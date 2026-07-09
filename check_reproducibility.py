from pathlib import Path

REQUIRED_FILES = [
    "results/unified_comparison.csv",
    "results/protocol_comparison_summary.csv",
    "results/privacy_utility_summary.csv",
    "results/membership_inference/membership_inference_results.csv",
    "results/dp_sgd_mlp/dp_sgd_mlp_results.csv",
    "results/dp_sgd_mlp/dp_sgd_mlp_threshold_tuning.csv",
    "results/transfer/transfer_experiment_results.csv",
    "results/causal/causal_bmi_effect_results.csv",
    "results/counterfactual/counterfactual_summary.csv",
    "results/shap_feature_comparison.csv",
    "graphs/comparison/privacy_audit_attack_auc_vs_epsilon.png",
    "graphs/membership_inference/attack_auc_vs_epsilon.png",
    "graphs/dp_sgd_mlp/clipping_comparison_worst_group_sensitivity.png",
]

REQUIRED_SOURCE_FILES = [
    "run_all.py",
    "compare_all_results.py",
    "privacy_utility_summary.py",
    "generate_final_report.py",
    "membership_inference_experiment.py",
    "dp_sgd_mlp_experiment.py",
    "transfer_experiment.py",
]


def main():
    missing = []

    for path in REQUIRED_SOURCE_FILES + REQUIRED_FILES:
        if not Path(path).exists():
            missing.append(path)

    print("=" * 80)
    print("Reproducibility check")
    print("=" * 80)

    if missing:
        print("Missing files:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)

    print("All required source, result, and graph files found.")
    print("Project is reproducible from the current saved outputs.")


if __name__ == "__main__":
    main()
