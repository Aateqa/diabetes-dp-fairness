import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from opacus import PrivacyEngine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)

from config import RESULTS_DIR, GRAPHS_DIR, RANDOM_STATE, TEST_SIZE
from data_loader_diabetes import load_raw_diabetes_data


FEATURE_SET_NAME = "Without Sensitive Attributes + Proxy-Reduced Features"
SENSITIVE_ATTRIBUTE = "sex_group"

BATCH_SIZE = 1024
N_EPOCHS = 20
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
MAX_GRAD_NORM = 1.0

EPSILONS = [0.5, 1.0, 3.0, 5.0, 10.0]
THRESHOLDS = [
    0.03, 0.05, 0.07, 0.10, 0.12,
    0.15, 0.18, 0.20, 0.22, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50
]


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DPMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return np.nan


def safe_brier(y_true, y_prob):
    try:
        return brier_score_loss(y_true, y_prob)
    except ValueError:
        return np.nan


def compute_group_metrics(y_true, y_pred, y_prob, sensitive_values):
    rows = []

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob)
    sensitive_values = np.asarray(sensitive_values)

    for group in sorted(pd.Series(sensitive_values).dropna().unique()):
        mask = sensitive_values == group

        group_y_true = y_true[mask]
        group_y_pred = y_pred[mask]
        group_y_prob = y_prob[mask]

        tp = ((group_y_true == 1) & (group_y_pred == 1)).sum()
        tn = ((group_y_true == 0) & (group_y_pred == 0)).sum()
        fp = ((group_y_true == 0) & (group_y_pred == 1)).sum()
        fn = ((group_y_true == 1) & (group_y_pred == 0)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        selection_rate = group_y_pred.mean() if len(group_y_pred) > 0 else np.nan

        rows.append({
            "group": group,
            "n_samples": int(mask.sum()),
            "selection_rate": selection_rate,
            "tpr": tpr,
            "fpr": fpr,
            "fnr": fnr,
            "auc": safe_auc(group_y_true, group_y_prob),
            "brier": safe_brier(group_y_true, group_y_prob),
        })

    group_df = pd.DataFrame(rows)

    if group_df.empty:
        return group_df, {}

    tpr_gap = group_df["tpr"].max() - group_df["tpr"].min()
    fpr_gap = group_df["fpr"].max() - group_df["fpr"].min()
    fnr_gap = group_df["fnr"].max() - group_df["fnr"].min()

    summary = {
        "dp_diff": group_df["selection_rate"].max() - group_df["selection_rate"].min(),
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "fnr_gap": fnr_gap,
        "equalized_odds_diff": max(tpr_gap, fpr_gap),
        "worst_group_sensitivity": group_df["tpr"].min(),
        "macro_avg_fnr": group_df["fnr"].mean(),
        "group_auc_mean": group_df["auc"].mean(),
        "group_brier_mean": group_df["brier"].mean(),
    }

    return group_df, summary


def predict_probabilities(model, X, device, batch_size=4096):
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    probs = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size].to(device)
            logits = model(batch)
            probs.extend(torch.sigmoid(logits).cpu().numpy())

    return np.asarray(probs)


def tune_threshold(y_true, y_prob):
    best_threshold = 0.25
    best_f1 = -1.0

    for threshold in THRESHOLDS:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold


def evaluate_model(method, epsilon, model, X_val, y_val, X_test, y_test, sensitive_test, device):
    val_prob = predict_probabilities(model, X_val, device)
    threshold = tune_threshold(y_val, val_prob)

    test_prob = predict_probabilities(model, X_test, device)
    test_pred = (test_prob >= threshold).astype(int)

    recall = recall_score(y_test, test_pred, zero_division=0)

    row = {
        "method": method,
        "epsilon": epsilon,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall": recall,
        "fnr": 1 - recall,
        "f1": f1_score(y_test, test_pred, zero_division=0),
        "auc": safe_auc(y_test, test_prob),
        "brier": safe_brier(y_test, test_prob),
    }

    group_df, fairness_summary = compute_group_metrics(
        y_true=y_test,
        y_pred=test_pred,
        y_prob=test_prob,
        sensitive_values=sensitive_test,
    )

    row.update(fairness_summary)

    pred_df = pd.DataFrame({
        "method": method,
        "epsilon": epsilon,
        "y_true": y_test,
        "y_prob": test_prob,
        "y_pred": test_pred,
        "sensitive_group": sensitive_test,
    })

    return row, group_df, pred_df


def make_pos_weight(y):
    y = np.asarray(y).astype(int)
    positives = y.sum()
    negatives = len(y) - positives

    if positives == 0:
        return torch.tensor(1.0, dtype=torch.float32)

    return torch.tensor(negatives / positives, dtype=torch.float32)


def train_non_private_mlp(X_train, y_train, device):
    model = DPMLP(input_dim=X_train.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pos_weight = make_pos_weight(y_train).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        losses = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if epoch == 1 or epoch % 5 == 0 or epoch == N_EPOCHS:
            print(f"Non-private MLP epoch {epoch:02d} | loss={np.mean(losses):.4f}")

    return model


def train_dp_sgd_mlp(X_train, y_train, target_epsilon, device):
    model = DPMLP(input_dim=X_train.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pos_weight = make_pos_weight(y_train).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    privacy_engine = PrivacyEngine(accountant="rdp")

    model, optimizer, private_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        epochs=N_EPOCHS,
        target_epsilon=target_epsilon,
        target_delta=1 / len(X_train),
        max_grad_norm=MAX_GRAD_NORM,
    )

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        losses = []

        for X_batch, y_batch in private_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if epoch == 1 or epoch % 5 == 0 or epoch == N_EPOCHS:
            spent_epsilon = privacy_engine.get_epsilon(delta=1 / len(X_train))
            print(
                f"DP-SGD MLP ε target={target_epsilon} | "
                f"epoch {epoch:02d} | "
                f"loss={np.mean(losses):.4f} | "
                f"spent ε={spent_epsilon:.4f}"
            )

    spent_epsilon = privacy_engine.get_epsilon(delta=1 / len(X_train))

    return model, spent_epsilon


def plot_results(results_df):
    os.makedirs(GRAPHS_DIR / "dp_sgd_mlp", exist_ok=True)

    metrics_to_plot = [
        "auc",
        "f1",
        "recall",
        "fnr",
        "dp_diff",
        "fnr_gap",
        "worst_group_sensitivity",
    ]

    x_labels = results_df["method"].astype(str)

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        plt.bar(x_labels, results_df[metric])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(metric)
        plt.title(f"DP-SGD MLP privacy-fairness tradeoff: {metric}")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        path = GRAPHS_DIR / "dp_sgd_mlp" / f"dp_sgd_mlp_{metric}.png"
        plt.savefig(path, dpi=300)
        plt.close()

        print(f"Saved: {path}")


def run_dp_sgd_mlp_experiment():
    set_seed(RANDOM_STATE)

    os.makedirs(RESULTS_DIR / "dp_sgd_mlp", exist_ok=True)
    os.makedirs(GRAPHS_DIR / "dp_sgd_mlp", exist_ok=True)

    print("\n" + "=" * 80)
    print("DP-SGD MLP experiment using Opacus")
    print("=" * 80)

    feature_sets, y, fairness_df, full_df = load_raw_diabetes_data(print_summary=False)

    X = feature_sets[FEATURE_SET_NAME].copy()
    y = y.astype(int).copy()
    sensitive = fairness_df[SENSITIVE_ATTRIBUTE].copy()

    X_trainval, X_test, y_trainval, y_test, s_trainval, s_test = train_test_split(
        X,
        y,
        sensitive,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
        X_trainval,
        y_trainval,
        s_trainval,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()
    y_test_arr = y_test.to_numpy()
    s_test_arr = s_test.to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Feature set: {FEATURE_SET_NAME}")
    print(f"Sensitive attribute: {SENSITIVE_ATTRIBUTE}")
    print(f"Train size: {len(X_train_sc):,}")
    print(f"Val size: {len(X_val_sc):,}")
    print(f"Test size: {len(X_test_sc):,}")
    print(f"Using device: {device}")

    all_rows = []
    all_groups = []
    all_predictions = []

    print("\n[1] Training non-private MLP baseline")
    baseline_model = train_non_private_mlp(
        X_train=X_train_sc,
        y_train=y_train_arr,
        device=device,
    )

    row, group_df, pred_df = evaluate_model(
        method="MLP non-private",
        epsilon=np.inf,
        model=baseline_model,
        X_val=X_val_sc,
        y_val=y_val_arr,
        X_test=X_test_sc,
        y_test=y_test_arr,
        sensitive_test=s_test_arr,
        device=device,
    )

    all_rows.append(row)
    group_df["method"] = "MLP non-private"
    group_df["epsilon"] = np.inf
    all_groups.append(group_df)
    all_predictions.append(pred_df)

    for epsilon in EPSILONS:
        print(f"\n[DP] Training DP-SGD MLP with target epsilon={epsilon}")

        model, spent_epsilon = train_dp_sgd_mlp(
            X_train=X_train_sc,
            y_train=y_train_arr,
            target_epsilon=epsilon,
            device=device,
        )

        row, group_df, pred_df = evaluate_model(
            method=f"DP-SGD MLP ε={epsilon}",
            epsilon=spent_epsilon,
            model=model,
            X_val=X_val_sc,
            y_val=y_val_arr,
            X_test=X_test_sc,
            y_test=y_test_arr,
            sensitive_test=s_test_arr,
            device=device,
        )

        row["target_epsilon"] = epsilon
        row["spent_epsilon"] = spent_epsilon

        all_rows.append(row)

        group_df["method"] = f"DP-SGD MLP ε={epsilon}"
        group_df["epsilon"] = spent_epsilon
        all_groups.append(group_df)

        all_predictions.append(pred_df)

    results_df = pd.DataFrame(all_rows)
    group_df = pd.concat(all_groups, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    results_path = RESULTS_DIR / "dp_sgd_mlp" / "dp_sgd_mlp_results.csv"
    group_path = RESULTS_DIR / "dp_sgd_mlp" / "dp_sgd_mlp_group_metrics.csv"
    pred_path = RESULTS_DIR / "dp_sgd_mlp" / "dp_sgd_mlp_predictions.csv"

    results_df.to_csv(results_path, index=False)
    group_df.to_csv(group_path, index=False)
    predictions_df.to_csv(pred_path, index=False)

    print("\n" + "=" * 80)
    print("DP-SGD MLP summary")
    print(
        results_df[
            [
                "method",
                "epsilon",
                "auc",
                "f1",
                "recall",
                "fnr",
                "dp_diff",
                "fnr_gap",
                "worst_group_sensitivity",
            ]
        ].round(4).to_string(index=False)
    )
    print("=" * 80)

    plot_results(results_df)

    print(f"\nSaved: {results_path}")
    print(f"Saved: {group_path}")
    print(f"Saved: {pred_path}")

    print("\nDP-SGD MLP experiment complete.")

    return results_df


if __name__ == "__main__":
    run_dp_sgd_mlp_experiment()