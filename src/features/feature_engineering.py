from __future__ import annotations

import numpy as np


def basic_window_stats(x_win: np.ndarray) -> np.ndarray:
    """
    Convert a (T, D) window into a compact stats vector.
    Useful for classical models and stage-2 classification.
    """
    if x_win.ndim != 2:
        raise ValueError("Expected x_win with shape (T, D)")

    mean = np.nanmean(x_win, axis=0)
    std = np.nanstd(x_win, axis=0)
    minv = np.nanmin(x_win, axis=0)
    maxv = np.nanmax(x_win, axis=0)
    q25 = np.nanpercentile(x_win, 25, axis=0)
    q50 = np.nanpercentile(x_win, 50, axis=0)
    q75 = np.nanpercentile(x_win, 75, axis=0)
    iqr = q75 - q25

    # Simple dynamics
    dx = np.diff(x_win, axis=0)
    dx_mean = np.nanmean(dx, axis=0) if dx.size else np.zeros_like(mean)
    dx_std = np.nanstd(dx, axis=0) if dx.size else np.zeros_like(mean)
    dx_abs_mean = np.nanmean(np.abs(dx), axis=0) if dx.size else np.zeros_like(mean)

    # Trend via least-squares slope per feature
    t = np.arange(x_win.shape[0], dtype=float)
    t = t - t.mean()
    denom = float((t * t).sum()) if x_win.shape[0] > 1 else 1.0
    xc = x_win - np.nanmean(x_win, axis=0, keepdims=True)
    slope = (t[:, None] * xc).sum(axis=0) / max(denom, 1e-12)

    # Temporal dependence and burstiness
    if x_win.shape[0] > 1:
        x0 = x_win[:-1]
        x1 = x_win[1:]
        num = np.nansum((x0 - np.nanmean(x0, axis=0)) * (x1 - np.nanmean(x1, axis=0)), axis=0)
        den = np.sqrt(
            np.nansum((x0 - np.nanmean(x0, axis=0)) ** 2, axis=0)
            * np.nansum((x1 - np.nanmean(x1, axis=0)) ** 2, axis=0)
        )
        autocorr1 = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    else:
        autocorr1 = np.zeros_like(mean)
    burst_ratio = np.divide(maxv - minv, std + 1e-6)

    return np.concatenate(
        [
            mean,
            std,
            minv,
            maxv,
            q25,
            q50,
            q75,
            iqr,
            dx_mean,
            dx_std,
            dx_abs_mean,
            slope,
            autocorr1,
            burst_ratio,
        ],
        axis=0,
    )

