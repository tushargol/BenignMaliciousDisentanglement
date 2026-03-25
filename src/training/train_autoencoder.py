from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ..models.autoencoder import Autoencoder


def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    input_dim: int,
    hidden_dim: int = 256,
    latent_dim: int = 64,
    batch_size: int = 256,
    lr: float = 1e-3,
    epochs: int = 30,
    device: str | None = None,
) -> Autoencoder:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float()),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float()),
        batch_size=batch_size,
        shuffle=False,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        tr = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += loss.item() * xb.size(0)

        model.eval()
        va = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                recon, _ = model(xb)
                loss = loss_fn(recon, xb)
                va += loss.item() * xb.size(0)

        tr /= max(1, len(train_loader.dataset))
        va /= max(1, len(val_loader.dataset))
        print(f"[AE] epoch={epoch:03d} train={tr:.5f} val={va:.5f}")

    return model

