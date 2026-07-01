import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
from models.tree_ensemble import make_xgb_model

from config import GRAPHS_DIR, RANDOM_STATE, TEST_SIZE
from data_loader_diabetes import load_diabetes_data

def plot_calibration_by_group(y_true, y_prob, group_values, group_name, output_path):
    plt.figure(figsize=(8, 6))

    colors = ["#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#000000", "#999999"]
    markers = ["o", "s", "^", "D", "x", "P"]

    for idx, group in enumerate(sorted(group_values.unique())):
        mask = group_values == group

        if mask.sum() < 50:
            print(f"Skipping calibration for {group_name}={group}: too few samples")
            continue

        group_y = y_true[mask]
        group_prob = y_prob[mask]

        if group_y.nunique() < 2:
            print(f"Skipping calibration for {group_name}={group}: only one class present")
            continue

        prob_true, prob_pred = calibration_curve(
            group_y,
            group_prob,
            n_bins=10,
            strategy="quantile",
        )

        plt.plot(
            prob_pred,
            prob_true,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=2,
            label=f"{group} (n={mask.sum()})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#666666",
        linewidth=1.5,
        label="Perfect calibration",
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed diabetes rate")
    plt.title(f"Diabetes Risk Prediction: Calibration Curve by {group_name}")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_roc_by_group(y_true, y_prob, group_values, group_name, output_path):
    plt.figure(figsize=(8, 6))

    colors = ["#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#000000", "#999999"]

    for idx, group in enumerate(sorted(group_values.unique())):
        mask = group_values == group

        if mask.sum() < 50:
            print(f"Skipping ROC for {group_name}={group}: too few samples")
            continue

        group_y = y_true[mask]
        group_prob = y_prob[mask]

        if group_y.nunique() < 2:
            print(f"Skipping ROC for {group_name}={group}: only one class present")
            continue

        fpr, tpr, _ = roc_curve(group_y, group_prob)
        group_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            color=colors[idx % len(colors)],
            linestyle="-",
            linewidth=2,
            label=f"{group} AUC={group_auc:.3f} (n={mask.sum()})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#666666",
        linewidth=1.5,
        label="Random classifier",
    )

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"Diabetes Risk Prediction: ROC Curve by {group_name}")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    feature_sets, y, fairness_df, df = load_diabetes_data(print_summary=False)

    # Use no-sensitive feature set for group evaluation:
    # the model does not directly see Sex or Age, but we still evaluate fairness by those groups.
    feature_set_name = "Without Sensitive Attributes"
    X = feature_sets[feature_set_name]

    print("\nRunning calibration and ROC analysis")
    print(f"Feature set: {feature_set_name}")
    print(f"X shape: {X.shape}")

    X_train, X_test, y_train, y_test, fairness_train, fairness_test = train_test_split(
        X,
        y,
        fairness_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_xgb_model(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    y_test = y_test.reset_index(drop=True)
    fairness_test = fairness_test.reset_index(drop=True)

    group_columns = {
        "sex_group": "Sex Group",
        "age_group": "Age Group",
        "income_group": "Income Group",
        "education_group": "Education Group",
        "intersection_group": "Intersection Group",
    }

    for col, label in group_columns.items():
        plot_calibration_by_group(
            y_true=y_test,
            y_prob=y_prob,
            group_values=fairness_test[col],
            group_name=label,
            output_path=f"{GRAPHS_DIR}/diabetes_xgb_no_sensitive_calibration_{col}.png",
        )

        plot_roc_by_group(
            y_true=y_test,
            y_prob=y_prob,
            group_values=fairness_test[col],
            group_name=label,
            output_path=f"{GRAPHS_DIR}/diabetes_xgb_no_sensitive_roc_{col}.png",
        )

    print("\nCalibration and ROC analysis complete.")


if __name__ == "__main__":
    main()