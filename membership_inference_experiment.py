"""
membership_inference_experiment.py

Membership inference attack against non-private and DP-SGD MLP models.

A membership inference attack asks: given a trained model and a sample, can an
adversary determine whether that sample was in the training set? This is a
concrete privacy vulnerability in healthcare ML — if an attacker knows a patient
record and can query the model, they may be able to infer that the patient
participated in the study.

We use a loss-based attack (Yeom et al., 2018): members (training samples) tend
to have lower loss than non-members (held-out test samples). The attack computes
per-sample loss and uses it as a membership score. Attack AUC measures how well
this distinguishes members from non-members:
    AUC = 1.0  ->  perfect attack, complete privacy violation
    AUC = 0.5  ->  random guessing, no information leaked

Differential privacy (Wang et al., 2019, ICML) provides a formal guarantee that
bounds this leakage as a function of epsilon. This experiment empirically
verifies that guarantee: as epsilon decreases (more privacy noise), attack AUC
should drop toward 0.5.
"""

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
from sklearn.metrics import roc_auc_score

from config import RESULTS_DIR, GRAPHS_DIR, RANDOM_STATE, TEST_SIZE
from data_loader_diabetes import load_raw_diabetes_data


FEATURE_SET_NAME = "Without Sensitive Attributes + Proxy-Reduced Features"
BATCH_SIZE = 1024
N_EPOCHS = 20
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
MAX_GRAD_NORM = 1.0
EPSILONS = [0.5, 1.0, 3.0, 5.0, 10.0]


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def make_pos_weight(y):
    y = np.asarray(y).astype(int)
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(neg / pos, dtype=torch.float32)


def train_non_private(X_train, y_train, device):
    model = MLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=make_pos_weight(y_train).to(device))
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for X_b, y_b in loader:
            optimizer.zero_grad()
            criterion(model(X_b.to(device)), y_b.to(device)).backward()
            optimizer.step()
        if epoch % 5 == 0 or epoch == N_EPOCHS:
            print(f"  Non-private epoch {epoch:02d}/{N_EPOCHS}")
    return model


def train_dp_sgd(X_train, y_train, target_epsilon, device):
    model = MLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=make_pos_weight(y_train).to(device))
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, private_loader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=loader,
        epochs=N_EPOCHS, target_epsilon=target_epsilon,
        target_delta=1 / len(X_train), max_grad_norm=MAX_GRAD_NORM,
    )
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for X_b, y_b in private_loader:
            optimizer.zero_grad()
            criterion(model(X_b.to(device)), y_b.to(device)).backward()
            optimizer.step()
        if epoch % 5 == 0 or epoch == N_EPOCHS:
            spent = privacy_engine.get_epsilon(delta=1 / len(X_train))
            print(f"  DP-SGD ε={target_epsilon} epoch {epoch:02d}/{N_EPOCHS} | spent ε={spent:.4f}")
    return model, privacy_engine.get_epsilon(delta=1 / len(X_train))


def compute_per_sample_loss(model, X, y, device, batch_size=4096):
    """
    Compute per-sample BCE loss. Members have lower loss on average than
    non-members — this is the signal the membership inference attack exploits.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    losses = []
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            X_b = X_t[i:i + batch_size].to(device)
            y_b = y_t[i:i + batch_size].to(device)
            loss = criterion(model(X_b), y_b)
            losses.extend(loss.cpu().numpy())
    return np.array(losses)


def run_attack(model, X_train, y_train, X_test, y_test, device):
    """
    Loss-based membership inference attack (Yeom et al., 2018).

    Lower loss = more likely to be a member. We negate loss so higher score
    means more likely member, then compute AUC against true membership labels.
    """
    member_losses     = compute_per_sample_loss(model, X_train, y_train, device)
    non_member_losses = compute_per_sample_loss(model, X_test,  y_test,  device)

    # Balance members and non-members for a fair attack AUC
    n = min(len(member_losses), len(non_member_losses))
    rng = np.random.default_rng(RANDOM_STATE)
    member_losses     = rng.choice(member_losses,     n, replace=False)
    non_member_losses = rng.choice(non_member_losses, n, replace=False)

    scores = np.concatenate([-member_losses, -non_member_losses])
    labels = np.concatenate([np.ones(n), np.zeros(n)])

    attack_auc = roc_auc_score(labels, scores)
    return attack_auc


def plot_attack_auc_vs_epsilon(results_df, output_dir):
    dp_df = results_df[results_df["target_epsilon"].notna()].copy()
    dp_df = dp_df.sort_values("target_epsilon")

    baseline_auc = results_df.loc[results_df["epsilon"] == "inf", "attack_auc"].values
    baseline_val = float(baseline_auc[0]) if len(baseline_auc) > 0 else None

    plt.figure(figsize=(9, 5))
    plt.plot(dp_df["target_epsilon"].astype(str), dp_df["attack_auc"],
             marker="o", color="steelblue", label="DP-SGD MLP")
    if baseline_val is not None:
        plt.axhline(baseline_val, color="tomato", linestyle="--",
                    label=f"Non-private MLP (AUC={baseline_val:.3f})")
    plt.axhline(0.5, color="gray", linestyle=":", label="Random guessing (AUC=0.5)")
    plt.xlabel("Privacy budget epsilon (lower = more private)")
    plt.ylabel("Membership inference attack AUC")
    plt.title(
        "Membership Inference Attack AUC vs Privacy Budget\n"
        "DP pushes attack toward random guessing (Wang et al., 2019, ICML)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "attack_auc_vs_epsilon.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def run_membership_inference_experiment():
    set_seed(RANDOM_STATE)

    result_dir = RESULTS_DIR / "membership_inference"
    graph_dir  = GRAPHS_DIR  / "membership_inference"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(graph_dir,  exist_ok=True)

    print("\n" + "=" * 80)
    print("Membership Inference Attack experiment")
    print("Connects to: Wang et al. (2019, ICML) -- DP-ERM bounds membership leakage")
    print("=" * 80)

    feature_sets, y, _, _ = load_raw_diabetes_data(print_summary=False)
    X = feature_sets[FEATURE_SET_NAME].copy()
    y = y.astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    y_train_arr = y_train.to_numpy()
    y_test_arr  = y_test.to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Feature set : {FEATURE_SET_NAME}")
    print(f"Train / test: {len(X_train_sc):,} / {len(X_test_sc):,}")
    print(f"Device      : {device}")

    rows = []

    # Non-private baseline
    print("\n[1] Non-private MLP")
    model = train_non_private(X_train_sc, y_train_arr, device)
    attack_auc = run_attack(model, X_train_sc, y_train_arr, X_test_sc, y_test_arr, device)
    rows.append({
        "method": "MLP non-private",
        "epsilon": "inf",
        "target_epsilon": np.nan,
        "spent_epsilon": np.nan,
        "attack_auc": attack_auc,
        "attack_advantage": attack_auc - 0.5,
    })
    print(f"  Attack AUC: {attack_auc:.4f}  (advantage: {attack_auc - 0.5:.4f})")

    # DP-SGD at each epsilon
    for epsilon in EPSILONS:
        print(f"\n[DP] DP-SGD MLP  epsilon={epsilon}")
        model, spent = train_dp_sgd(X_train_sc, y_train_arr, epsilon, device)
        attack_auc = run_attack(model, X_train_sc, y_train_arr, X_test_sc, y_test_arr, device)
        rows.append({
            "method": f"DP-SGD MLP ε={epsilon}",
            "epsilon": str(spent),
            "target_epsilon": float(epsilon),
            "spent_epsilon": float(spent),
            "attack_auc": attack_auc,
            "attack_advantage": attack_auc - 0.5,
        })
        print(f"  Spent ε={spent:.4f} | Attack AUC: {attack_auc:.4f}  (advantage: {attack_auc - 0.5:.4f})")

    results_df = pd.DataFrame(rows)
    results_path = result_dir / "membership_inference_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 80)
    print("Membership inference summary")
    print(results_df.round(4).to_string(index=False))
    print("=" * 80)

    plot_attack_auc_vs_epsilon(results_df, graph_dir)

    print(f"\nSaved: {results_path}")
    print("\n  Theoretical note:")
    print("  DP-ERM (Wang et al., 2019, ICML) guarantees that the advantage of any")
    print("  membership inference adversary is bounded by O(epsilon / n^0.5).")
    print("  The empirical attack AUC should trend toward 0.5 as epsilon decreases,")
    print("  but near-random attack AUCs should be interpreted cautiously: they may")
    print("  indicate either strong privacy or simply weak attack power on this run.")

    print("\nMembership inference experiment complete.")
    return results_df


if __name__ == "__main__":
    run_membership_inference_experiment()
