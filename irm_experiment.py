import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import RESULTS_DIR, GRAPHS_DIR, RANDOM_STATE
from data_loader_diabetes import load_raw_diabetes_data
from metrics import compute_fairness_metrics, compute_group_rates, safe_auc, safe_brier


FEATURE_SET_NAME = "Without Sensitive Attributes + Proxy-Reduced Features"
ENVIRONMENT_COLUMN = "income_group"
TARGET_ENVIRONMENT = "low_income"
SOURCE_ENVIRONMENTS = ["high_income", "middle_income"]

N_EPOCHS = 120
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
IRM_LAMBDA = 100.0
IRM_ANNEAL_EPOCH = 30
WEIGHT_DECAY = 1e-4

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


class IRMMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.network(x).view(-1)




def predict_probabilities(model, X, device, batch_size=4096):
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    probs = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size].to(device)
            logits = model(batch)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            probs.extend(batch_probs)

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


def evaluate_method(method_name, model, X_val, y_val, X_test, y_test, sensitive_test, device):
    val_prob = predict_probabilities(model, X_val, device)
    threshold = tune_threshold(y_val, val_prob)

    test_prob = predict_probabilities(model, X_test, device)
    test_pred = (test_prob >= threshold).astype(int)

    recall = recall_score(y_test, test_pred, zero_division=0)

    row = {
        "method": method_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall": recall,
        "fnr": 1 - recall,
        "f1": f1_score(y_test, test_pred, zero_division=0),
        "auc": safe_auc(y_test, test_prob),
        "brier": safe_brier(y_test, test_prob),
    }

    fairness_summary = compute_fairness_metrics(
        y_true=y_test,
        y_pred=test_pred,
        y_prob=test_prob,
        protected_values=sensitive_test,
    )
    group_df = compute_group_rates(
        y_true=y_test,
        y_pred=test_pred,
        y_prob=test_prob,
        group_values=sensitive_test,
    )

    row.update(fairness_summary)

    prediction_df = pd.DataFrame({
        "method": method_name,
        "y_true": y_test,
        "y_prob": test_prob,
        "y_pred": test_pred,
        "sensitive_group": sensitive_test,
    })

    return row, group_df, prediction_df


def make_pos_weight(y):
    y = np.asarray(y).astype(int)
    positives = y.sum()
    negatives = len(y) - positives

    if positives == 0:
        return torch.tensor(1.0, dtype=torch.float32)

    return torch.tensor(negatives / positives, dtype=torch.float32)


def train_erm(X_train, y_train, device, input_dim, method_name):
    model = IRMMLP(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    pos_weight = make_pos_weight(y_train).to(device)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()

        optimizer.zero_grad()
        logits = model(X_tensor)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            y_tensor,
            pos_weight=pos_weight,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch == 1 or epoch % 20 == 0 or epoch == N_EPOCHS:
            print(f"{method_name} epoch {epoch:03d} | loss={loss.item():.4f}")

    return model


def irm_penalty(logits, y, pos_weight, device):
    scale = torch.tensor(1.0, device=device, requires_grad=True)

    loss = F.binary_cross_entropy_with_logits(
        logits * scale,
        y,
        pos_weight=pos_weight,
    )

    grad = torch.autograd.grad(
        loss,
        [scale],
        create_graph=True,
    )[0]

    return torch.sum(grad ** 2)


def train_irm(environment_data, device, input_dim):
    model = IRMMLP(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    env_tensors = {}
    all_y = []

    for env_name, (X_env, y_env) in environment_data.items():
        env_tensors[env_name] = (
            torch.tensor(X_env, dtype=torch.float32).to(device),
            torch.tensor(y_env, dtype=torch.float32).to(device),
        )
        all_y.extend(y_env)

    pos_weight = make_pos_weight(np.asarray(all_y)).to(device)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        env_losses = []
        env_penalties = []

        for env_name, (X_env, y_env) in env_tensors.items():
            logits = model(X_env)

            loss = F.binary_cross_entropy_with_logits(
                logits,
                y_env,
                pos_weight=pos_weight,
            )

            penalty = irm_penalty(
                logits=logits,
                y=y_env,
                pos_weight=pos_weight,
                device=device,
            )

            env_losses.append(loss)
            env_penalties.append(penalty)

        mean_loss = torch.stack(env_losses).mean()
        mean_penalty = torch.stack(env_penalties).mean()

        penalty_weight = 1.0 if epoch < IRM_ANNEAL_EPOCH else IRM_LAMBDA
        objective = mean_loss + penalty_weight * mean_penalty

        if penalty_weight > 1.0:
            objective = objective / penalty_weight

        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch == 1 or epoch % 20 == 0 or epoch == N_EPOCHS:
            print(
                f"IRM epoch {epoch:03d} | "
                f"loss={mean_loss.item():.4f} | "
                f"penalty={mean_penalty.item():.6f} | "
                f"weight={penalty_weight:.1f}"
            )

    return model


def plot_irm_results(results_df, output_path):
    metrics_to_plot = [
        "auc",
        "f1",
        "recall",
        "fnr",
        "dp_diff",
        "fnr_gap",
        "worst_group_sensitivity",
        "macro_avg_fnr",
    ]

    plot_df = results_df[["method"] + metrics_to_plot].copy()

    for metric in metrics_to_plot:
        plt.figure(figsize=(9, 6))
        plt.bar(plot_df["method"], plot_df[metric])
        plt.ylabel(metric)
        plt.xlabel("Method")
        plt.title(f"IRM transfer comparison: {metric}")
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        metric_output = output_path.parent / f"irm_{metric}.png"
        plt.savefig(metric_output, dpi=300)
        plt.close()

        print(f"Saved: {metric_output}")


def run_irm_experiment():
    os.makedirs(RESULTS_DIR / "irm", exist_ok=True)
    os.makedirs(GRAPHS_DIR / "irm", exist_ok=True)

    print("\n" + "=" * 80)
    print("Invariant Risk Minimization experiment")
    print("=" * 80)
    print(f"Feature set     : {FEATURE_SET_NAME}")
    print(f"Source envs     : {SOURCE_ENVIRONMENTS}")
    print(f"Target env      : {TARGET_ENVIRONMENT}")
    print("Target fairness : sex_group within low_income target domain")
    print("=" * 80)

    feature_sets, y, fairness_df, full_df = load_raw_diabetes_data(print_summary=False)

    X = feature_sets[FEATURE_SET_NAME].copy()
    y = y.astype(int).copy()
    env = fairness_df[ENVIRONMENT_COLUMN].copy()
    sensitive = fairness_df["sex_group"].copy()

    source_mask = env.isin(SOURCE_ENVIRONMENTS)
    target_mask = env == TARGET_ENVIRONMENT

    X_source = X.loc[source_mask]
    y_source = y.loc[source_mask]
    env_source = env.loc[source_mask]

    X_target = X.loc[target_mask]
    y_target = y.loc[target_mask]
    sensitive_target = sensitive.loc[target_mask]

    print(f"Source size: {len(X_source):,}")
    print(f"Target size: {len(X_target):,}")
    print(f"Source diabetes rate: {y_source.mean():.4f}")
    print(f"Target diabetes rate: {y_target.mean():.4f}")

    source_train_idx, source_val_idx = train_test_split(
        X_source.index,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_source,
    )

    target_trainval_idx, target_test_idx = train_test_split(
        X_target.index,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_target,
    )

    target_train_idx, target_val_idx = train_test_split(
        target_trainval_idx,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y.loc[target_trainval_idx],
    )

    scaler = StandardScaler()
    scaler.fit(X.loc[source_train_idx])

    X_source_train = scaler.transform(X.loc[source_train_idx])
    y_source_train = y.loc[source_train_idx].to_numpy()

    X_source_val = scaler.transform(X.loc[source_val_idx])
    y_source_val = y.loc[source_val_idx].to_numpy()

    X_target_train = scaler.transform(X.loc[target_train_idx])
    y_target_train = y.loc[target_train_idx].to_numpy()

    X_target_val = scaler.transform(X.loc[target_val_idx])
    y_target_val = y.loc[target_val_idx].to_numpy()

    X_target_test = scaler.transform(X.loc[target_test_idx])
    y_target_test = y.loc[target_test_idx].to_numpy()
    sensitive_target_test = sensitive.loc[target_test_idx].to_numpy()

    set_seed(RANDOM_STATE)

    input_dim = X_source_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    results = []
    all_group_results = []
    all_predictions = []

    print("\n[1/3] ERM source-pooled baseline")
    erm_model = train_erm(
        X_train=X_source_train,
        y_train=y_source_train,
        device=device,
        input_dim=input_dim,
        method_name="ERM",
    )

    row, group_df, pred_df = evaluate_method(
        method_name="ERM source-pooled",
        model=erm_model,
        X_val=X_source_val,
        y_val=y_source_val,
        X_test=X_target_test,
        y_test=y_target_test,
        sensitive_test=sensitive_target_test,
        device=device,
    )
    results.append(row)
    group_df["method"] = "ERM source-pooled"
    all_group_results.append(group_df)
    all_predictions.append(pred_df)

    print("\n[2/3] IRM across source income environments")

    environment_data = {}

    for env_name in SOURCE_ENVIRONMENTS:
        env_idx = [
            idx for idx in source_train_idx
            if env.loc[idx] == env_name
        ]

        environment_data[env_name] = (
            scaler.transform(X.loc[env_idx]),
            y.loc[env_idx].to_numpy(),
        )

        print(
            f"  {env_name}: n={len(env_idx):,}, "
            f"diabetes rate={y.loc[env_idx].mean():.4f}"
        )

    irm_model = train_irm(
        environment_data=environment_data,
        device=device,
        input_dim=input_dim,
    )

    row, group_df, pred_df = evaluate_method(
        method_name="IRM source-invariant",
        model=irm_model,
        X_val=X_source_val,
        y_val=y_source_val,
        X_test=X_target_test,
        y_test=y_target_test,
        sensitive_test=sensitive_target_test,
        device=device,
    )
    results.append(row)
    group_df["method"] = "IRM source-invariant"
    all_group_results.append(group_df)
    all_predictions.append(pred_df)

    print("\n[3/3] Oracle target-trained reference")
    oracle_model = train_erm(
        X_train=X_target_train,
        y_train=y_target_train,
        device=device,
        input_dim=input_dim,
        method_name="Oracle",
    )

    row, group_df, pred_df = evaluate_method(
        method_name="Oracle target-trained",
        model=oracle_model,
        X_val=X_target_val,
        y_val=y_target_val,
        X_test=X_target_test,
        y_test=y_target_test,
        sensitive_test=sensitive_target_test,
        device=device,
    )
    results.append(row)
    group_df["method"] = "Oracle target-trained"
    all_group_results.append(group_df)
    all_predictions.append(pred_df)

    results_df = pd.DataFrame(results)
    group_results_df = pd.concat(all_group_results, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    results_path = RESULTS_DIR / "irm" / "irm_experiment_results.csv"
    group_path = RESULTS_DIR / "irm" / "irm_group_metrics.csv"
    predictions_path = RESULTS_DIR / "irm" / "irm_predictions.csv"

    results_df.to_csv(results_path, index=False)
    group_results_df.to_csv(group_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\n" + "=" * 80)
    print("IRM summary")
    print(
        results_df[
            [
                "method",
                "auc",
                "f1",
                "recall",
                "fnr",
                "dp_diff",
                "fnr_gap",
                "worst_group_sensitivity",
                "macro_avg_fnr",
            ]
        ].round(4).to_string(index=False)
    )
    print("=" * 80)

    plot_irm_results(
        results_df=results_df,
        output_path=GRAPHS_DIR / "irm" / "irm_results.png",
    )

    print(f"\nSaved: {results_path}")
    print(f"Saved: {group_path}")
    print(f"Saved: {predictions_path}")
    print("\nIRM experiment complete.")

    return results_df


if __name__ == "__main__":
    run_irm_experiment()
