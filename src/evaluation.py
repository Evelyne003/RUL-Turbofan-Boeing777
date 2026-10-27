"""Módulo 4 — Métricas (RMSE, MAE, función de puntuación asimétrica) y gráficas.

Función de puntuación (Saxena et al., 2008), con d = RUL_estimada − RUL_real:

    s = Σ ( exp(−d/13) − 1 )  si d < 0   (estimación temprana, se subestima la RUL)
    s = Σ ( exp( d/10) − 1 )  si d ≥ 0   (estimación tardía, se SOBREestima la RUL)

La penalización es asimétrica: las estimaciones tardías (d > 0) pesan más porque implican
operar un motor más allá de su vida segura. Menor es mejor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import MODELS_DIR, SUBSETS  # noqa: E402


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_pred - y_true)))


def nasa_score(y_true, y_pred) -> float:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def regression_report(y_true, y_pred) -> dict:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "score": nasa_score(y_true, y_pred),
        "n": int(len(d)),
        "pct_estimacion_tardia": float(np.mean(d > 0) * 100.0),  # d>0: se sobreestima la RUL
    }


#gráficas
def plot_predictions(y_true, y_pred, title: str = "", path: Path | None = None):
    """Izquierda: RUL real vs. estimada ordenando los motores por RUL real (formato habitual
    en la literatura). Derecha: dispersión con línea ideal y = x."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    order = np.argsort(y_true)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(y_true[order], "-", color="tab:blue", label="RUL real")
    ax[0].plot(y_pred[order], "o", ms=3.5, color="tab:red", alpha=0.8, label="RUL estimada")
    ax[0].set(xlabel="Motores de prueba (ordenados por RUL real)", ylabel="RUL (ciclos)",
              title=title or "RUL real vs. estimada")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax[1].scatter(y_true, y_pred, s=14, alpha=0.7)
    ax[1].plot(lim, lim, "k--", lw=1, label="Ideal (y = x)")
    ax[1].set(xlabel="RUL real (ciclos)", ylabel="RUL estimada (ciclos)",
              title="Dispersión", xlim=lim, ylim=lim)
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def plot_error_distribution(y_true, y_pred, path: Path | None = None):
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(d, bins=30, color="tab:purple", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set(xlabel="Error = RUL estimada − RUL real (ciclos)", ylabel="Frecuencia",
           title="Distribución del error (d > 0: estimación tardía)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def plot_training_history(history: pd.DataFrame, path: Path | None = None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["loss"], label="entrenamiento")
    ax[0].plot(history["val_loss"], label="validación")
    ax[0].set(xlabel="Época", ylabel="MSE", title="Pérdida"); ax[0].legend(); ax[0].grid(alpha=0.3)
    if "learning_rate" in history:
        ax[1].semilogy(history["learning_rate"])
        ax[1].set(xlabel="Época", ylabel="Tasa de aprendizaje", title="ReduceLROnPlateau")
        ax[1].grid(alpha=0.3)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def save_metrics(metrics: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


# reevaluación de un modelo
def evaluate_saved(subset: str, models_dir: Path | None = None, raw_dir: Path | None = None) -> dict:
    """Reevalúa un modelo guardado en el conjunto de prueba del subconjunto indicado."""
    from src.inference import load_artifacts, predict_last_windows
    from src.ingestion import load_subset

    art = load_artifacts(subset, models_dir)
    data = load_subset(subset, raw_dir)
    preds, _ = predict_last_windows(art, data.test)
    cap = art.config["rul_cap"]
    return {
        "test": regression_report(data.rul_test, preds),
        "test_capped_labels": regression_report(np.minimum(data.rul_test, cap), preds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reevaluación de un modelo guardado")
    parser.add_argument("--subset", default="FD001", choices=SUBSETS)
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    args = parser.parse_args()
    res = evaluate_saved(args.subset, Path(args.models_dir))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()