from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ..models.classifier import AnomalyClassifier


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    hidden_dim: int = 128,
    sample_weights_train: np.ndarray | None = None,
    sample_weights_val: np.ndarray | None = None,
    batch_size: int = 256,
    lr: float = 1e-3,
    epochs: int = 30,
    device: str | None = None,
) -> AnomalyClassifier:
    """
    y_* must be binary: 0=benign, 1=malicious.
    Optional per-sample weights can emphasize specific attack families.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AnomalyClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    # Per-sample loss; we aggregate manually to support sample weights.
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    if sample_weights_train is None:
        w_train = np.ones_like(y_train, dtype=np.float32)
    else:
        w_train = sample_weights_train.astype(np.float32)
    if sample_weights_val is None:
        w_val = np.ones_like(y_val, dtype=np.float32)
    else:
        w_val = sample_weights_val.astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
            torch.from_numpy(w_train).float(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float(),
            torch.from_numpy(w_val).float(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        tr = 0.0
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            logits = model(xb)
            loss_raw = loss_fn(logits, yb)
            loss = (loss_raw * wb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += loss.item() * xb.size(0)

        model.eval()
        va = 0.0
        with torch.no_grad():
            for xb, yb, wb in val_loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                logits = model(xb)
                loss_raw = loss_fn(logits, yb)
                loss = (loss_raw * wb).mean()
                va += loss.item() * xb.size(0)

        tr /= max(1, len(train_loader.dataset))
        va /= max(1, len(val_loader.dataset))
        print(f"[CLF] epoch={epoch:03d} train={tr:.5f} val={va:.5f}")

    return model

