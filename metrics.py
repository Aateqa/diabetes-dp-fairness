import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def compute_group_rates(y_true, y_pred, group_values):
    """
    Computes selection rate, TPR, FPR, FNR, and accuracy per group.
    """
    rows = []

    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    group_values = pd.Series(group_values).reset_index(drop=True)

    for group in sorted(group_values.unique()):
        mask = group_values == group

        group_y_true = y_true[mask]
        group_y_pred = y_pred[mask]

        tp = ((group_y_true == 1) & (group_y_pred == 1)).sum()
        tn = ((group_y_true == 0) & (group_y_pred == 0)).sum()
        fp = ((group_y_true == 0) & (group_y_pred == 1)).sum()
        fn = ((group_y_true == 1) & (group_y_pred == 0)).sum()

        selection_rate = group_y_pred.mean()
        tpr = safe_divide(tp, tp + fn)
        fpr = safe_divide(fp, fp + tn)
        fnr = safe_divide(fn, fn + tp)
        accuracy = accuracy_score(group_y_true, group_y_pred)

        rows.append({
            "group": group,
            "n_samples": int(mask.sum()),
            "selection_rate": selection_rate,
            "tpr": tpr,
            "fpr": fpr,
            "fnr": fnr,
            "accuracy": accuracy,
        })

    return pd.DataFrame(rows)


def compute_fairness_metrics(y_true, y_pred, protected_values):
    """
    Computes fairness gaps for one protected attribute.
    """
    group_rates = compute_group_rates(y_true, y_pred, protected_values)

    selection_rates = group_rates["selection_rate"].dropna()
    tprs = group_rates["tpr"].dropna()
    fprs = group_rates["fpr"].dropna()
    fnrs = group_rates["fnr"].dropna()
    accuracies = group_rates["accuracy"].dropna()

    dp_diff = selection_rates.max() - selection_rates.min()
    tpr_gap = tprs.max() - tprs.min()
    fpr_gap = fprs.max() - fprs.min()
    fnr_gap = fnrs.max() - fnrs.min()
    accuracy_gap = accuracies.max() - accuracies.min()

    if selection_rates.max() == 0:
        di_ratio = np.nan
    else:
        di_ratio = selection_rates.min() / selection_rates.max()

    equalized_odds_diff = max(tpr_gap, fpr_gap)

    return {
        "dp_diff": dp_diff,
        "di_ratio": di_ratio,
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "fnr_gap": fnr_gap,
        "accuracy_gap": accuracy_gap,
        "equalized_odds_diff": equalized_odds_diff,
    }


def compute_all_metrics(y_true, y_pred, y_prob, fairness_df):
    """
    Computes utility metrics and fairness metrics for all protected attributes.
    """
    results = {}

    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["precision"] = precision_score(y_true, y_pred, zero_division=0)
    results["recall"] = recall_score(y_true, y_pred, zero_division=0)
    results["f1"] = f1_score(y_true, y_pred, zero_division=0)

    try:
        results["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        results["auc"] = np.nan

    results["error_rate"] = 1 - results["accuracy"]

    for protected_attr in fairness_df.columns:
        fairness_metrics = compute_fairness_metrics(
            y_true=y_true,
            y_pred=y_pred,
            protected_values=fairness_df[protected_attr],
        )

        for metric_name, metric_value in fairness_metrics.items():
            results[f"{protected_attr}_{metric_name}"] = metric_value

    return results


def append_metric_dict(results_list, model_name, feature_set_name, y_true, y_pred, y_prob, fairness_df):
    metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        fairness_df=fairness_df,
    )

    row = {
        "model": model_name,
        "feature_set": feature_set_name,
    }

    row.update(metrics)
    results_list.append(row)