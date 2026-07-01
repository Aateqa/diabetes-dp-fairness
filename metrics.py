import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)


def counterfactual_fairness_ratio(model, X, flip_fn):
    """
    Fraction of individuals whose prediction changes under a demographic intervention.

    Operationalises counterfactual fairness (Kusner et al., 2017): a model is
    counterfactually fair if its prediction is unchanged in the counterfactual
    world where the sensitive attribute took a different value.

    Lower = more counterfactually fair.
    """
    X_cf = flip_fn(X)
    if X_cf is None:
        return np.nan
    original_pred = model.predict(X)
    cf_pred = model.predict(X_cf)
    return float((original_pred != cf_pred).mean())


def get_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return np.nan


def safe_brier(y_true, y_prob):
    try:
        return brier_score_loss(y_true, y_prob)
    except ValueError:
        return np.nan


def compute_group_rates(y_true, y_pred, y_prob, group_values, min_group_size=500):
    """
    Computes selection rate, TPR, FPR, FNR, accuracy, AUC, and Brier score per group.

    Groups with fewer than min_group_size rows are warned because fairness metrics
    may be unstable for small groups.
    """
    rows = []

    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    y_prob = pd.Series(y_prob).reset_index(drop=True)
    group_values = pd.Series(group_values).reset_index(drop=True)

    for group in sorted(group_values.dropna().unique()):
        mask = group_values == group
        n_samples = int(mask.sum())

        group_y_true = y_true[mask]
        group_y_pred = y_pred[mask]
        group_y_prob = y_prob[mask]

        if n_samples < min_group_size:
            print(
                f"WARNING: group '{group}' has only {n_samples} rows. "
                "Fairness metrics may be unstable."
            )

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
            "n_samples": n_samples,
            "selection_rate": selection_rate,
            "tpr": tpr,
            "fpr": fpr,
            "fnr": fnr,
            "accuracy": accuracy,
            "auc": safe_auc(group_y_true, group_y_prob),
            "brier": safe_brier(group_y_true, group_y_prob),
        })

    return pd.DataFrame(rows)


def compute_fairness_metrics(y_true, y_pred, y_prob, protected_values):
    group_rates = compute_group_rates(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        group_values=protected_values,
    )

    selection_rates = group_rates["selection_rate"].dropna()
    tprs = group_rates["tpr"].dropna()
    fprs = group_rates["fpr"].dropna()
    fnrs = group_rates["fnr"].dropna()
    accuracies = group_rates["accuracy"].dropna()
    aucs = group_rates["auc"].dropna()
    briers = group_rates["brier"].dropna()

    dp_diff = selection_rates.max() - selection_rates.min()
    tpr_gap = tprs.max() - tprs.min()
    fpr_gap = fprs.max() - fprs.min()
    fnr_gap = fnrs.max() - fnrs.min()
    accuracy_gap = accuracies.max() - accuracies.min()

    if selection_rates.max() == 0:
        di_ratio = np.nan
    else:
        di_ratio = selection_rates.min() / selection_rates.max()

    # Non-standard definition used in this project:
    # equalized_odds_diff = max(tpr_gap, fpr_gap)
    equalized_odds_diff = max(tpr_gap, fpr_gap)

    return {
        "dp_diff": dp_diff,
        "di_ratio": di_ratio,
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "fnr_gap": fnr_gap,
        "accuracy_gap": accuracy_gap,
        "equalized_odds_diff": equalized_odds_diff,
        "worst_group_sensitivity": tprs.min(),
        "macro_avg_fnr": fnrs.mean(),
        "group_auc_mean": aucs.mean() if len(aucs) > 0 else np.nan,
        "group_brier_mean": briers.mean() if len(briers) > 0 else np.nan,
    }


def compute_all_metrics(y_true, y_pred, y_prob, fairness_df):
    results = {}

    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["precision"] = precision_score(y_true, y_pred, zero_division=0)
    results["recall"] = recall_score(y_true, y_pred, zero_division=0)
    results["f1"] = f1_score(y_true, y_pred, zero_division=0)

    results["auc"] = safe_auc(y_true, y_prob)
    results["brier"] = safe_brier(y_true, y_prob)
    results["error_rate"] = 1 - results["accuracy"]
    results["fnr"] = 1 - results["recall"]

    for protected_attr in fairness_df.columns:
        fairness_metrics = compute_fairness_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            protected_values=fairness_df[protected_attr],
        )

        for metric_name, metric_value in fairness_metrics.items():
            results[f"{protected_attr}_{metric_name}"] = metric_value

    return results


def append_metric_dict(
    results_list,
    model_name,
    feature_set_name,
    y_true,
    y_pred,
    y_prob,
    fairness_df,
):
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