import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


class DiabetesTensorDataset(Dataset):
    def __init__(self, X, y, s):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.s = torch.tensor(s, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.s[idx]


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


class VFAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        latent_dim=16,
        grl_lambda=1.0,
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


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )


class VFAEClassifier:
    """
    sklearn-compatible wrapper around VFAE for use in cross-validation pipelines.

    Handles its own internal train/val split, scaling, and threshold tuning so it
    can be dropped into the same cross-validation loop as sklearn estimators.
    Uses n_epochs=20 by default (vs 30 in vfae_experiment.py) for CV speed.
    """

    def __init__(
        self,
        hidden_dim=64,
        latent_dim=16,
        grl_lambda=1.0,
        n_epochs=20,
        batch_size=1024,
        learning_rate=1e-3,
        recon_weight=0.10,
        kl_weight=0.001,
        adv_weight=0.50,
        patience=5,
        val_fraction=0.15,
        random_state=42,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.grl_lambda = grl_lambda
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

        self._scaler = None
        self._model = None
        self._device = None
        self._threshold = 0.25

    def get_params(self, deep=True):
        return {
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "grl_lambda": self.grl_lambda,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "recon_weight": self.recon_weight,
            "kl_weight": self.kl_weight,
            "adv_weight": self.adv_weight,
            "patience": self.patience,
            "val_fraction": self.val_fraction,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _map_sensitive(self, s):
        s = pd.Series(s) if not isinstance(s, pd.Series) else s
        mapped = s.map({"female": 0, "male": 1})
        if mapped.isna().any():
            return np.asarray(s, dtype=int)
        return mapped.astype(int).values

    def fit(self, X, y, sensitive_features=None):
        from metrics import safe_auc

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        s_arr = (
            self._map_sensitive(sensitive_features).astype(np.float32)
            if sensitive_features is not None
            else np.zeros(len(y_arr), dtype=np.float32)
        )

        X_tr, X_val, y_tr, y_val, s_tr, s_val = train_test_split(
            X_arr, y_arr, s_arr,
            test_size=self.val_fraction,
            random_state=self.random_state,
            stratify=y_arr,
        )

        self._scaler = StandardScaler()
        X_tr_sc = self._scaler.fit_transform(X_tr)
        X_val_sc = self._scaler.transform(X_val)

        from torch.utils.data import DataLoader

        train_loader = DataLoader(
            DiabetesTensorDataset(X_tr_sc, y_tr, s_tr),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._model = VFAE(
            input_dim=X_tr_sc.shape[1],
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            grl_lambda=self.grl_lambda,
        ).to(self._device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        clf_loss = nn.BCEWithLogitsLoss()
        adv_loss = nn.BCEWithLogitsLoss()
        recon_loss = nn.MSELoss()

        best_auc = -np.inf
        best_state = None
        patience_counter = 0

        for _ in range(self.n_epochs):
            self._model.train()
            for X_b, y_b, s_b in train_loader:
                X_b = X_b.to(self._device)
                y_b = y_b.to(self._device)
                s_b = s_b.to(self._device)
                optimizer.zero_grad()
                x_recon, y_logit, s_logit, mu, logvar, _ = self._model(X_b)
                loss = (
                    clf_loss(y_logit, y_b)
                    + self.adv_weight * adv_loss(s_logit, s_b)
                    + self.recon_weight * recon_loss(x_recon, X_b)
                    + self.kl_weight * kl_divergence(mu, logvar)
                )
                loss.backward()
                optimizer.step()

            val_probs = self._infer(X_val_sc)
            val_auc = safe_auc(y_val, val_probs)

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = copy.deepcopy(self._model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        # Tune threshold on val set using F1.
        val_probs = self._infer(X_val_sc)
        best_f1, best_t = -1.0, 0.25
        for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            preds = (val_probs >= t).astype(int)
            f = f1_score(y_val, preds, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        self._threshold = best_t

        return self

    def _infer(self, X_scaled):
        self._model.eval()
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                batch = X_t[i:i + self.batch_size].to(self._device)
                _, y_logit, _, _, _, _ = self._model(batch)
                probs.extend(torch.sigmoid(y_logit).cpu().numpy().reshape(-1))
        return np.array(probs)

    def predict_proba(self, X):
        X_sc = self._scaler.transform(np.asarray(X, dtype=np.float32))
        probs = self._infer(X_sc)
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        X_sc = self._scaler.transform(np.asarray(X, dtype=np.float32))
        probs = self._infer(X_sc)
        return (probs >= self._threshold).astype(int)
