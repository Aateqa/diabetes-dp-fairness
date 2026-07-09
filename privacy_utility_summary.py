from pathlib import Path
import pandas as pd
import numpy as np

RESULTS_DIR = Path("results")
OUT_PATH = RESULTS_DIR / "privacy_utility_summary.csv"


def load_csv(path):
    path = Path(path)
    if not path.exists():
        print(f"Missing: {path}")
        return None
    return pd.read_csv(path)


def normalise_dp_sgd_method_name(method):
    method = str(method)
    method = method.replace("DP-SGD MLP ε=", "DP-SGD MLP epsilon=")
    return method


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    dp_sgd = load_csv("results/dp_sgd_mlp/dp_sgd_mlp_results.csv")
    mi = load_csv("results/membership_inference/membership_inference_results.csv")

    if dp_sgd is None or mi is None:
        raise SystemExit("Required result files are missing. Run the experiments first.")

    rows = []

    # Non-private baseline
    non_private_utility = dp_sgd[dp_sgd["method"].astype(str).str.contains("non-private", case=False, na=False)]
    non_private_attack = mi[mi["method"].astype(str).str.contains("non-private", case=False, na=False)]

    if len(non_private_utility) > 0 and len(non_private_attack) > 0:
        u = non_private_utility.iloc[0]
        a = non_private_attack.iloc[0]
        rows.append({
            "model": "MLP non-private",
            "epsilon": "inf",
            "clipping": "none",
            "classification_auc": u.get("auc", np.nan),
            "recall": u.get("recall", np.nan),
            "worst_group_sensitivity": u.get("worst_group_sensitivity", np.nan),
            "fnr_gap": u.get("fnr_gap", np.nan),
            "dp_diff": u.get("dp_diff", np.nan),
            "membership_attack_auc": a.get("attack_auc", np.nan),
            "membership_attack_advantage": a.get("attack_advantage", np.nan),
            "attack_feature_used": a.get("attack_feature_used", np.nan),
        })

    # DP-SGD models.
    # Membership audit was run for epsilon 0.5, 2.0, 10.0, 50.0.
    # Utility sweep has 0.5, 1.0, 3.0, 5.0, 10.0 with fixed/adaptive clipping.
    for _, u in dp_sgd.iterrows():
        method = str(u.get("method", ""))
        if "DP-SGD" not in method:
            continue

        eps = float(u.get("epsilon"))
        clipping = str(u.get("clipping", ""))

        attack_match = mi[np.isclose(pd.to_numeric(mi["target_epsilon"], errors="coerce"), eps, equal_nan=False)]

        if len(attack_match) > 0:
            a = attack_match.iloc[0]
            attack_auc = a.get("attack_auc", np.nan)
            attack_advantage = a.get("attack_advantage", np.nan)
            attack_feature = a.get("attack_feature_used", np.nan)
        else:
            attack_auc = np.nan
            attack_advantage = np.nan
            attack_feature = np.nan

        rows.append({
            "model": method,
            "epsilon": eps,
            "clipping": clipping,
            "classification_auc": u.get("auc", np.nan),
            "recall": u.get("recall", np.nan),
            "worst_group_sensitivity": u.get("worst_group_sensitivity", np.nan),
            "fnr_gap": u.get("fnr_gap", np.nan),
            "dp_diff": u.get("dp_diff", np.nan),
            "membership_attack_auc": attack_auc,
            "membership_attack_advantage": attack_advantage,
            "attack_feature_used": attack_feature,
        })

    summary = pd.DataFrame(rows)

    # Sort: non-private first, then DP epsilon/clipping.
    summary["_eps_sort"] = pd.to_numeric(summary["epsilon"], errors="coerce").fillna(np.inf)
    summary["_clip_sort"] = summary["clipping"].map({"none": 0, "fixed": 1, "adaptive": 2}).fillna(9)
    summary = summary.sort_values(["_eps_sort", "_clip_sort"]).drop(columns=["_eps_sort", "_clip_sort"])

    summary.to_csv(OUT_PATH, index=False)

    print("=" * 80)
    print("Privacy-utility summary")
    print("=" * 80)
    print(summary.to_string(index=False))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
