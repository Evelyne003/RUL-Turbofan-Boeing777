from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from src.config import (DATA_RAW, MODELS_DIR, N_REGIMES, SUBSETS, TrainConfig)
from src.evaluation import (plot_error_distribution, plot_predictions, plot_training_history,
                            regression_report, save_metrics)
from src.ingestion import load_subset
from src.model import build_from_config, make_callbacks
from src.preprocessing import (Preprocessor, add_rul, build_test_windows,
                               build_training_windows, split_by_engine)

logger = logging.getLogger(__name__)


@dataclass
class DataBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray          # RUL real (sin tope) del último ciclo de cada motor de prueba
    test_units: np.ndarray
    preprocessor: Preprocessor
    feature_cols: list
    train_units: np.ndarray
    val_units: np.ndarray


def prepare_data(cfg: TrainConfig, raw_dir: Path | None = None) -> DataBundle:
    """Módulos 1 y 2. El preprocesador se ajusta únicamente con los motores de
    entrenamiento (80 %); validación y prueba no influyen en ninguna estadística."""
    data = load_subset(cfg.subset, raw_dir if raw_dir is not None else DATA_RAW)
    train = add_rul(data.train, cap=cfg.rul_cap)
    tr_units, val_units = split_by_engine(train["unit"].unique(), cfg.val_fraction, cfg.seed)

    pre = Preprocessor(n_regimes=N_REGIMES[cfg.subset], seed=cfg.seed)
    pre.fit(train[train["unit"].isin(tr_units)])
    train_n, test_n = pre.transform(train), pre.transform(data.test)
    feats = pre.feature_cols_

    X_tr, y_tr, _ = build_training_windows(train_n[train_n["unit"].isin(tr_units)], feats, cfg.window)
    X_va, y_va, _ = build_training_windows(train_n[train_n["unit"].isin(val_units)], feats, cfg.window)
    X_te, test_units = build_test_windows(test_n, feats, cfg.window)
    if len(X_te) != len(data.rul_test):
        raise ValueError("Número de motores de prueba distinto al de RUL_FD00X.txt")
    return DataBundle(X_tr, y_tr, X_va, y_va, X_te, data.rul_test, test_units, pre, feats,
                      tr_units, val_units)


def train_model(cfg: TrainConfig, bundle: DataBundle, verbose: int = 0):
    keras.utils.set_random_seed(cfg.seed)
    model = build_from_config(cfg, n_features=len(bundle.feature_cols))
    hist = model.fit(
        bundle.X_train, bundle.y_train,
        validation_data=(bundle.X_val, bundle.y_val),
        epochs=cfg.epochs, batch_size=cfg.batch_size,
        callbacks=make_callbacks(cfg), shuffle=True, verbose=verbose,
    )
    return model, pd.DataFrame(hist.history)


def run(cfg: TrainConfig, raw_dir: Path | None = None, out_dir: Path | None = None,
        verbose: int = 0) -> dict:
    """Ejecuta el flujo completo y guarda artefactos en ``models/<subset>/``."""
    out = Path(out_dir) if out_dir is not None else MODELS_DIR
    out = out / cfg.subset
    out.mkdir(parents=True, exist_ok=True)

    bundle = prepare_data(cfg, raw_dir)
    logger.info("%s | train=%s val=%s test=%s | %d sensores",
                cfg.subset, bundle.X_train.shape, bundle.X_val.shape, bundle.X_test.shape,
                len(bundle.feature_cols))
    model, history = train_model(cfg, bundle, verbose=verbose)

    val_pred = model.predict(bundle.X_val, verbose=0).ravel()
    test_pred = np.clip(model.predict(bundle.X_test, verbose=0).ravel(), 0, cfg.rul_cap)
    metrics = {
        "subset": cfg.subset,
        "validation_windows": regression_report(bundle.y_val, val_pred),
        # Evaluación contra la RUL real sin tope (RUL_FD00X.txt)
        "test": regression_report(bundle.y_test, test_pred),
        # Misma evaluación con la RUL real topada en cap (convención de algunos trabajos)
        "test_capped_labels": regression_report(np.minimum(bundle.y_test, cfg.rul_cap), test_pred),
        "epochs_run": int(len(history)),
        "features": bundle.feature_cols,
        "dropped_sensors": bundle.preprocessor.dropped_cols_,
    }

    model.save(out / "model.keras")
    bundle.preprocessor.save(out / "preprocessor.joblib")
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    save_metrics(metrics, out / "metrics.json")
    history.to_csv(out / "history.csv", index=False)
    pd.DataFrame({"motor": bundle.test_units, "rul_real": bundle.y_test,
                  "rul_estimada": test_pred}).to_csv(out / "test_predictions.csv", index=False)

    fig_dir = out / "figures"
    plot_predictions(bundle.y_test, test_pred, f"{cfg.subset} — prueba", fig_dir / "pred_vs_real.png")
    plot_error_distribution(bundle.y_test, test_pred, fig_dir / "error_dist.png")
    plot_training_history(history, fig_dir / "training_history.png")
    import matplotlib.pyplot as plt
    plt.close("all")
    return metrics


def parse_args(argv=None) -> argparse.Namespace:
    d = TrainConfig()
    p = argparse.ArgumentParser(description="Entrenamiento y evaluación LSTM para RUL")
    p.add_argument("--subset", default=d.subset, choices=SUBSETS)
    p.add_argument("--window", type=int, default=d.window)
    p.add_argument("--rul-cap", type=int, default=d.rul_cap)
    p.add_argument("--val-fraction", type=float, default=d.val_fraction)
    p.add_argument("--lstm-units", type=int, nargs="+", default=list(d.lstm_units))
    p.add_argument("--dense-units", type=int, nargs="+", default=list(d.dense_units))
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--learning-rate", type=float, default=d.learning_rate)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--raw-dir", default=str(DATA_RAW))
    p.add_argument("--out-dir", default=str(MODELS_DIR))
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    a = parse_args(argv)
    cfg = TrainConfig(
        subset=a.subset, window=a.window, rul_cap=a.rul_cap, val_fraction=a.val_fraction,
        lstm_units=tuple(a.lstm_units), dense_units=tuple(a.dense_units), dropout=a.dropout,
        learning_rate=a.learning_rate, epochs=a.epochs, batch_size=a.batch_size, seed=a.seed,
    )
    m = run(cfg, Path(a.raw_dir), Path(a.out_dir), verbose=a.verbose)
    print(json.dumps({k: m[k] for k in ("subset", "test", "test_capped_labels")}, indent=2))


if __name__ == "__main__":
    main()