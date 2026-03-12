from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class FitResult:
    model_name: str
    params: np.ndarray
    covariance: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        model = get_model_function(self.model_name)
        return model(x, *self.params)


def load_msa(path: str | Path, delimiter: str = ",", skiprows: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Load 2-column .msa file (energy, counts)."""
    arr = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in MSA file, got shape {arr.shape}.")
    return arr[:, 0], arr[:, 1]


def crop_window(x: np.ndarray, y: np.ndarray, start_edge: float, end_edge: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = (x > start_edge) & (x < end_edge)
    return x[mask], y[mask]


def model_exp1(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.exp(b * x)


def model_exp2(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a * np.exp(b * x) + c * np.exp(d * x)


def model_power1(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.power(x, b)


def model_power2(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.power(x, b) + c


def get_model_function(name: str) -> Callable[..., np.ndarray]:
    mapping = {
        "exp1": model_exp1,
        "exp2": model_exp2,
        "power1": model_power1,
        "power2": model_power2,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported model '{name}'. Choose from: {', '.join(mapping)}")
    return mapping[name]


def default_initial_guess(name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y_span = max(float(np.max(y) - np.min(y)), 1.0)
    y_min = float(np.min(y))
    if name == "exp1":
        return np.array([float(np.max(y)), -0.01])
    if name == "exp2":
        return np.array([float(np.max(y)), -0.01, y_span / 2.0, -0.001])
    if name == "power1":
        x_med = max(float(np.median(x)), 1.0)
        return np.array([float(np.max(y)) / x_med, -1.0])
    if name == "power2":
        x_med = max(float(np.median(x)), 1.0)
        return np.array([float(np.max(y)) / x_med, -1.0, y_min])
    raise ValueError(name)


def fit_background(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    exclude_above: float | None = None,
    p0: np.ndarray | None = None,
    maxfev: int = 100000,
) -> tuple[FitResult, np.ndarray, np.ndarray]:
    """Fit background model and return fit object + full-window fit & residual arrays."""
    model = get_model_function(model_name)

    if exclude_above is None:
        fit_mask = np.ones_like(x, dtype=bool)
    else:
        fit_mask = x < exclude_above

    x_fit = x[fit_mask]
    y_fit = y[fit_mask]

    if p0 is None:
        p0 = default_initial_guess(model_name, x_fit, y_fit)

    params, covariance = curve_fit(model, x_fit, y_fit, p0=p0, maxfev=maxfev)

    fit_result = FitResult(model_name=model_name, params=params, covariance=covariance)
    y_model_full = fit_result.predict(x)
    residuals = y - y_model_full
    return fit_result, y_model_full, residuals
