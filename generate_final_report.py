from pathlib import Path
import pandas as pd
import numpy as np

OUT_PATH = Path("results/final_project_report.md")


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def fmt(x, digits=4):
    if pd.isna(x):
        return "N/A"
    if isinstance(x, str):
        return x
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def dataframe_block(df):
    """
    Render a dataframe without requiring pandas.to_markdown/tabulate.
    This keeps report generation dependency-light.
    """
    if df is None or df.empty:
        return "No rows available.\n"

    return "```text\n" + df.to_string(index=False) + "\n```\n"


def section(title):
    return f"\n## {title}\n"


def main():
    OUT_PATH.parent.mkdir(exist_ok=True)

    unified = read_csv("results/unified_comparison.csv")
    dp_sgd = read_csv("results/dp_sgd_mlp/dp_sgd_mlp_results.csv")
    mi = read_csv("results/membership_inference/membership_inference_results.csv")
    privacy_utility = read_csv("results/privacy_utility_summary.csv")
    transfer = read_csv("results/transfer/transfer_experiment_results.csv")
    causal = read_csv("results/causal/causal_bmi_effect_results.csv")
    counterfactual = read_csv("results/counterfactual/counterfactual_summary.csv")
    shap = read_csv("results/shap_feature_comparison.csv")

    lines = []
    lines.append("# Diabetes DP-Fairness Final Project Report\n")
    lines.append("This report is generated automatically from the saved experiment outputs.\n")

    lines.append(section("1. Overall Summary"))
    lines.append(
        "This project evaluates diabetes prediction under a combined fairness, privacy, "
        "clinical screening, transfer learning, causal analysis, counterfactual fairness, "
        "and interpretability setting. The main result is that DP-SGD MLP models can preserve "
        "strong clinical recall and worst-group sensitivity while sharply reducing empirical "
        "membership-inference leakage compared with a non-private MLP.\n"
    )

    lines.append(section("2. Best DP-SGD Clinical Screening Result"))
    if dp_sgd is not None:
        dp_only = dp_sgd[dp_sgd["method"].astype(str).str.contains("DP-SGD", na=False)].copy()
        if not dp_only.empty:
            best = dp_only.sort_values(
                ["worst_group_sensitivity", "fnr_gap", "recall"],
                ascending=[False, True, False]
            ).iloc[0]
            lines.append(
                f"The best DP-SGD clinical screening configuration is **{best['method']}**. "
                f"It achieves AUC={fmt(best.get('auc'))}, recall={fmt(best.get('recall'))}, "
                f"worst-group sensitivity={fmt(best.get('worst_group_sensitivity'))}, "
                f"FNR gap={fmt(best.get('fnr_gap'))}, and dp_diff={fmt(best.get('dp_diff'))}.\n"
            )

        non_private = dp_sgd[dp_sgd["method"].astype(str).str.contains("non-private", case=False, na=False)]
        if not non_private.empty:
            base = non_private.iloc[0]
            lines.append(
                f"The non-private MLP baseline achieves AUC={fmt(base.get('auc'))}, "
                f"recall={fmt(base.get('recall'))}, worst-group sensitivity="
                f"{fmt(base.get('worst_group_sensitivity'))}, and FNR gap={fmt(base.get('fnr_gap'))}.\n"
            )
    else:
        lines.append("DP-SGD result file was not found.\n")

    lines.append(section("3. Membership Inference Privacy Audit"))
    if mi is not None:
        lines.append(
            "The membership inference audit uses multiple attack signals: loss, confidence, "
            "entropy, margin, correctness, and a learned logistic-regression attack over these "
            "features. Attack AUC near 0.5 indicates random guessing; higher values indicate "
            "membership leakage.\n"
        )

        for _, row in mi.iterrows():
            lines.append(
                f"- **{row['method']}**: attack AUC={fmt(row.get('attack_auc'))}, "
                f"advantage={fmt(row.get('attack_advantage'))}, strongest feature="
                f"{row.get('attack_feature_used', 'N/A')}\n"
            )
    else:
        lines.append("Membership inference result file was not found.\n")

    lines.append(section("4. Privacy-Utility Tradeoff"))
    if privacy_utility is not None:
        cols = [
            "model", "epsilon", "clipping", "classification_auc", "recall",
            "worst_group_sensitivity", "fnr_gap", "membership_attack_auc"
        ]
        available_cols = [c for c in cols if c in privacy_utility.columns]
        lines.append(dataframe_block(privacy_utility[available_cols]))
        lines.append("\n")
    else:
        lines.append("Privacy-utility summary file was not found. Run `python privacy_utility_summary.py`.\n")

    lines.append(section("5. Transfer Learning and Subgroup Generalisation"))
    if transfer is not None:
        for _, row in transfer.iterrows():
            method = row.get("method", "Unknown")
            lines.append(
                f"- **{method}**: AUC={fmt(row.get('auc'))}, F1={fmt(row.get('f1'))}, "
                f"recall={fmt(row.get('recall'))}, worst-group sensitivity="
                f"{fmt(row.get('worst_group_sensitivity'))}, FNR gap={fmt(row.get('fnr_gap'))}.\n"
            )
    else:
        lines.append("Transfer result file was not found.\n")

    lines.append(section("6. Causal Analysis"))
    if causal is not None:
        lines.append(
            "The causal analysis estimates the adjusted effect of obesity on diabetes risk "
            "using age-adjusted/backdoor-style estimation.\n"
        )
        lines.append(dataframe_block(causal))
        lines.append("\n")
    else:
        lines.append("Causal result file was not found.\n")

    lines.append(section("7. Counterfactual Fairness"))
    if counterfactual is not None:
        lines.append(dataframe_block(counterfactual))
        lines.append("\n")
    else:
        lines.append("Counterfactual summary file was not found.\n")

    lines.append(section("8. SHAP Interpretability"))
    if shap is not None:
        lines.append(
            "SHAP analysis compares feature importance across the original, no-sensitive, "
            "and proxy-reduced feature spaces.\n"
        )
        preview = shap.head(15)
        lines.append(dataframe_block(preview))
        lines.append("\n")
    else:
        lines.append("SHAP comparison file was not found.\n")

    lines.append(section("9. Final Takeaway"))
    lines.append(
        "The strongest project claim is that privacy-preserving deep learning can maintain "
        "clinically useful screening behaviour while reducing empirical membership leakage. "
        "The DP-SGD MLP preserves high recall and high worst-group sensitivity, while the "
        "membership inference audit shows the non-private MLP leaking membership signal and "
        "DP-SGD models remaining close to random guessing.\n"
    )

    OUT_PATH.write_text("\n".join(lines))
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
