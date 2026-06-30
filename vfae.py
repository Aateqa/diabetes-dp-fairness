import os
import random
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

from config import (
    RESULTS_DIR,
    GRAPHS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)

from data_loader_diabetes import load_raw_diabetes_data

# Configuration

SENSITIVE_ATTRIBUTE = "sex_group"
FEATURE_SET_NAME = "Without Sensitive Attributes + Proxy-Reduced Features"

N_EPOCHS = 30
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
LATENT_DIM = 16
HIDDEN_DIM = 64

RECON_WEIGHT = 0.10
KL_WEIGHT = 0.001
ADV_WEIGHT = 0.50
GRL_LAMBDA = 1.0

PATIENCE = 5

# Thresholds for screening-style evaluation.
# Lower thresholds usually improve recall and reduce false negatives.
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
 
# Reproducibility

def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Dataset

class DiabetesTensorDataset(Dataset):
    def __init__(self, X, y, s):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.s = torch.tensor(s, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.s[idx]

# Gradient reversal layer

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_value):
        ctx.lambda_value = lambda_value
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_value * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_value=1.0):
        super().__init__()
        self.lambda_value = lambda_value

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_value)

# VFAE-style model

class VFAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        grl_lambda=GRL_LAMBDA,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.gradient_reversal = GradientReversalLayer(lambda_value=grl_lambda)

        self.sensitive_adversary = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        x_recon = self.decoder(z)
        y_logit = self.classifier(z)

        reversed_z = self.gradient_reversal(z)
        s_logit = self.sensitive_adversary(reversed_z)

        return x_recon, y_logit, s_logit, mu, logvar, z

# Loss and metrics

def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )


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

    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    y_prob = pd.Series(y_prob).reset_index(drop=True)
    sensitive_values = pd.Series(sensitive_values).reset_index(drop=True)

    for group in sorted(sensitive_values.unique()):
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
        selection_rate = group_y_pred.mean()

        auc = safe_auc(group_y_true, group_y_prob)
        brier = safe_brier(group_y_true, group_y_prob)

        rows.append({
            "group": group,
            "n_samples": int(mask.sum()),
            "selection_rate": selection_rate,
            "tpr": tpr,
            "fpr": fpr,
            "fnr": fnr,
            "auc": auc,
            "brier": brier,
        })

    group_df = pd.DataFrame(rows)

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


def evaluate_model(model, data_loader, device, threshold=0.5):
    model.eval()

    all_y = []
    all_s = []
    all_probs = []
    all_preds = []
    all_z = []

    with torch.no_grad():
        for X_batch, y_batch, s_batch in data_loader:
            X_batch = X_batch.to(device)

            _, y_logit, _, _, _, z = model(X_batch)

            y_prob = torch.sigmoid(y_logit).cpu().numpy().reshape(-1)
            y_pred = (y_prob >= threshold).astype(int)

            all_y.extend(y_batch.numpy().reshape(-1))
            all_s.extend(s_batch.numpy().reshape(-1))
            all_probs.extend(y_prob)
            all_preds.extend(y_pred)
            all_z.append(z.cpu().numpy())

    y_true = np.array(all_y).astype(int)
    sensitive = np.array(all_s).astype(int)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    z_matrix = np.vstack(all_z)

    recall = recall_score(y_true, y_pred, zero_division=0)

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall,
        "fnr": 1 - recall,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": safe_auc(y_true, y_prob),
        "brier": safe_brier(y_true, y_prob),
    }

    group_df, fairness_summary = compute_group_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        sensitive_values=sensitive,
    )

    metrics.update(fairness_summary)

    # Screening-oriented score:
    # Higher is better.
    # Prioritises high worst-group sensitivity and penalises high FNR / FNR gap.
    metrics["clinical_fairness_score"] = (
        metrics["worst_group_sensitivity"]
        - metrics["macro_avg_fnr"]
        - metrics["fnr_gap"]
    )

    return metrics, group_df, z_matrix, y_true, sensitive, y_prob, y_pred


def tune_thresholds(model, val_loader, device):
    """
    Evaluates VFAE across multiple thresholds and selects the best threshold
    for screening-oriented fairness.

    We prioritise:
    - high worst-group sensitivity
    - low macro-averaged FNR
    - low FNR gap
    """
    rows = []

    for threshold in THRESHOLDS:
        metrics, _, _, _, _, _, _ = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            threshold=threshold,
        )

        rows.append(metrics)

    threshold_df = pd.DataFrame(rows)

    ranked_threshold_df = threshold_df.sort_values(
        by=[
            "clinical_fairness_score",
            "worst_group_sensitivity",
            "recall",
            "auc",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    best_threshold = float(ranked_threshold_df.loc[0, "threshold"])

    return best_threshold, threshold_df, ranked_threshold_df

# Training

def train_vfae(model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    classification_loss_fn = nn.BCEWithLogitsLoss()
    sensitive_loss_fn = nn.BCEWithLogitsLoss()
    reconstruction_loss_fn = nn.MSELoss()

    history = []

    best_val_auc = -np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()

        epoch_losses = []

        for X_batch, y_batch, s_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            s_batch = s_batch.to(device)

            optimizer.zero_grad()

            x_recon, y_logit, s_logit, mu, logvar, _ = model(X_batch)

            classification_loss = classification_loss_fn(y_logit, y_batch)
            adversarial_loss = sensitive_loss_fn(s_logit, s_batch)
            reconstruction_loss = reconstruction_loss_fn(x_recon, X_batch)
            kl_loss = kl_divergence(mu, logvar)

            total_loss = (
                classification_loss
                + ADV_WEIGHT * adversarial_loss
                + RECON_WEIGHT * reconstruction_loss
                + KL_WEIGHT * kl_loss
            )

            total_loss.backward()
            optimizer.step()

            epoch_losses.append(total_loss.item())

        val_metrics, _, _, _, _, _, _ = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            threshold=0.5,
        )

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_auc": val_metrics["auc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_recall": val_metrics["recall"],
            "val_fnr": val_metrics["fnr"],
            "val_dp_diff": val_metrics["dp_diff"],
            "val_fnr_gap": val_metrics["fnr_gap"],
            "val_worst_group_sensitivity": val_metrics["worst_group_sensitivity"],
            "val_macro_avg_fnr": val_metrics["macro_avg_fnr"],
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={row['train_loss']:.4f} | "
            f"val_auc={row['val_auc']:.4f} | "
            f"val_recall={row['val_recall']:.4f} | "
            f"val_dp_diff={row['val_dp_diff']:.4f} | "
            f"val_worst_group_sens={row['val_worst_group_sensitivity']:.4f}"
        )

        if row["val_auc"] > best_val_auc:
            best_val_auc = row["val_auc"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history)

# Plots

def plot_training_history(history_df, output_path):
    plt.figure(figsize=(10, 6))

    plt.plot(
        history_df["epoch"],
        history_df["val_auc"],
        marker="o",
        label="Validation AUC",
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_recall"],
        marker="o",
        label="Validation recall",
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_worst_group_sensitivity"],
        marker="o",
        label="Worst-group sensitivity",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("VFAE Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved training plot: {output_path}")


def plot_threshold_tuning(threshold_df, output_path):
    plot_df = threshold_df.sort_values("threshold").copy()

    plt.figure(figsize=(10, 6))

    plt.plot(
        plot_df["threshold"],
        plot_df["recall"],
        marker="o",
        label="Recall",
    )

    plt.plot(
        plot_df["threshold"],
        plot_df["worst_group_sensitivity"],
        marker="o",
        label="Worst-group sensitivity",
    )

    plt.plot(
        plot_df["threshold"],
        plot_df["macro_avg_fnr"],
        marker="o",
        label="Macro-averaged FNR",
    )

    plt.plot(
        plot_df["threshold"],
        plot_df["fnr_gap"],
        marker="o",
        label="FNR gap",
    )

    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.title("VFAE Threshold Tuning for Diabetes Screening")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved threshold tuning plot: {output_path}")


def plot_vfae_metrics(metrics_df, output_path):
    selected = metrics_df[
        [
            "auc",
            "recall",
            "fnr",
            "dp_diff",
            "fnr_gap",
            "worst_group_sensitivity",
            "macro_avg_fnr",
        ]
    ].T

    selected.columns = ["value"]

    plt.figure(figsize=(10, 6))
    plt.bar(selected.index, selected["value"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Metric value")
    plt.title("VFAE Test Metrics After Threshold Tuning")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved metrics plot: {output_path}")

# Main experiment

def run_vfae_experiment():
    set_seed(RANDOM_STATE)

    os.makedirs(RESULTS_DIR / "vfae", exist_ok=True)
    os.makedirs(GRAPHS_DIR / "vfae", exist_ok=True)

    print("\n" + "=" * 80)
    print("Running VFAE experiment")
    print("=" * 80)

    feature_sets, y, fairness_df, full_df = load_raw_diabetes_data(print_summary=False)

    X = feature_sets[FEATURE_SET_NAME].copy()
    y = y.astype(int).copy()

    sensitive_raw = fairness_df[SENSITIVE_ATTRIBUTE].copy()

    sensitive_binary = sensitive_raw.map({
        "female": 0,
        "male": 1,
    })

    if sensitive_binary.isna().any():
        raise ValueError(
            f"Sensitive attribute {SENSITIVE_ATTRIBUTE} could not be mapped cleanly. "
            f"Unique values found: {sorted(sensitive_raw.dropna().unique())}"
        )

    sensitive_binary = sensitive_binary.astype(int)

    print(f"Feature set: {FEATURE_SET_NAME}")
    print(f"Sensitive attribute: {SENSITIVE_ATTRIBUTE}")
    print(f"X shape: {X.shape}")

    X_train_val, X_test, y_train_val, y_test, s_train_val, s_test = train_test_split(
        X,
        y,
        sensitive_binary,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
        X_train_val,
        y_train_val,
        s_train_val,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = DiabetesTensorDataset(
        X=X_train_scaled,
        y=y_train.to_numpy(),
        s=s_train.to_numpy(),
    )

    val_dataset = DiabetesTensorDataset(
        X=X_val_scaled,
        y=y_val.to_numpy(),
        s=s_val.to_numpy(),
    )

    test_dataset = DiabetesTensorDataset(
        X=X_test_scaled,
        y=y_test.to_numpy(),
        s=s_test.to_numpy(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VFAE(
        input_dim=X_train_scaled.shape[1],
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        grl_lambda=GRL_LAMBDA,
    ).to(device)

    model, history_df = train_vfae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    print("\nTuning VFAE decision threshold on validation set...")

    best_threshold, threshold_df, ranked_threshold_df = tune_thresholds(
        model=model,
        val_loader=val_loader,
        device=device,
    )

    print("\nThreshold tuning results:")
    print(threshold_df.round(4))

    print("\nThreshold tuning ranking:")
    print(ranked_threshold_df.round(4))

    print(f"\nBest threshold selected: {best_threshold:.2f}")

    test_metrics, group_df, z_matrix, y_true, sensitive, y_prob, y_pred = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        threshold=best_threshold,
    )

    metrics_df = pd.DataFrame([test_metrics])

    print("\nVFAE test metrics after threshold tuning:")
    print(metrics_df.round(4).T)

    print("\nVFAE group metrics after threshold tuning:")
    print(group_df.round(4))

    history_path = RESULTS_DIR / "vfae" / "vfae_training_history.csv"
    threshold_path = RESULTS_DIR / "vfae" / "vfae_threshold_tuning.csv"
    ranked_threshold_path = RESULTS_DIR / "vfae" / "vfae_threshold_tuning_ranked.csv"
    metrics_path = RESULTS_DIR / "vfae" / "vfae_test_metrics.csv"
    group_path = RESULTS_DIR / "vfae" / "vfae_group_metrics.csv"
    predictions_path = RESULTS_DIR / "vfae" / "vfae_test_predictions.csv"
    latent_path = RESULTS_DIR / "vfae" / "vfae_latent_embeddings.npy"
    model_path = RESULTS_DIR / "vfae" / "vfae_model.pt"

    history_df.to_csv(history_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    ranked_threshold_df.to_csv(ranked_threshold_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    group_df.to_csv(group_path, index=False)

    predictions_df = pd.DataFrame({
        "y_true": y_true,
        "sensitive_sex": sensitive,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "threshold": best_threshold,
    })

    predictions_df.to_csv(predictions_path, index=False)

    np.save(latent_path, z_matrix)

    torch.save(model.state_dict(), model_path)

    plot_training_history(
        history_df=history_df,
        output_path=GRAPHS_DIR / "vfae" / "vfae_training_history.png",
    )

    plot_threshold_tuning(
        threshold_df=threshold_df,
        output_path=GRAPHS_DIR / "vfae" / "vfae_threshold_tuning.png",
    )

    plot_vfae_metrics(
        metrics_df=metrics_df,
        output_path=GRAPHS_DIR / "vfae" / "vfae_test_metrics.png",
    )

    print("\nSaved:")
    print(history_path)
    print(threshold_path)
    print(ranked_threshold_path)
    print(metrics_path)
    print(group_path)
    print(predictions_path)
    print(latent_path)
    print(model_path)

    print("\nVFAE experiment complete.")

    return metrics_df, group_df


if __name__ == "__main__":
    run_vfae_experiment()