# Inferencia y lógica de alertas 
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.preprocessing import Preprocessor, build_test_windows, sliding_windows

# Umbrales de alerta por defecto (ciclos). NO provienen de la literatura: son parámetros de
# política de mantenimiento que debe definir el usuario final
DEFAULT_WARNING = 60
DEFAULT_CRITICAL = 30


@dataclass
class Artifacts:
    model: keras.Model
    preprocessor: Preprocessor
    config: dict
    metrics: dict | None = None


def load_artifacts(subset: str, models_dir: Path | None = None) -> Artifacts:
    d = Path(models_dir) if models_dir is not None else MODELS_DIR
    d = d / subset
    if not (d / "model.keras").exists():
        raise FileNotFoundError(f"No hay modelo entrenado en {d}. Ejecute `make train SUBSET={subset}`.")
    model = keras.models.load_model(d / "model.keras", compile=False)
    pre = Preprocessor.load(d / "preprocessor.joblib")
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    mpath = d / "metrics.json"
    metrics = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else None
    return Artifacts(model=model, preprocessor=pre, config=cfg, metrics=metrics)


def _clip(pred: np.ndarray, cap: float) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=float).ravel(), 0.0, float(cap))


def predict_last_windows(art: Artifacts, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Estima la RUL del último ciclo observado de cada motor. Devuelve (rul, unidades)."""
    pp = art.preprocessor.transform(test_df)
    X, units = build_test_windows(pp, art.preprocessor.feature_cols_, art.config["window"])
    pred = art.model.predict(X, verbose=0)
    return _clip(pred, art.config["rul_cap"]), units


def predict_engine_curve(art: Artifacts, engine_df: pd.DataFrame) -> pd.DataFrame:
    engine_df = engine_df.sort_values("cycle")
    pp = art.preprocessor.transform(engine_df)
    X = sliding_windows(pp[art.preprocessor.feature_cols_].to_numpy(),
                        art.config["window"], pad_start=True)
    pred = art.model.predict(X, verbose=0)
    return pd.DataFrame({"cycle": pp["cycle"].to_numpy(),
                         "rul_pred": _clip(pred, art.config["rul_cap"])})


def true_rul_curve(n_cycles: int, rul_final: float, cap: float | None = None) -> np.ndarray:
    """RUL real en cada ciclo de un motor de prueba truncado: rul_final + (n − t)."""
    t = np.arange(1, n_cycles + 1)
    rul = rul_final + (n_cycles - t)
    return np.minimum(rul, cap) if cap is not None else rul.astype(float)


def rul_status(rul: float, warning: float = DEFAULT_WARNING, critical: float = DEFAULT_CRITICAL) -> str:
    if rul <= critical:
        return "CRÍTICO"
    if rul <= warning:
        return "PRECAUCIÓN"
    return "NORMAL"


def fleet_status(art: Artifacts, test_df: pd.DataFrame,
                 warning: float = DEFAULT_WARNING, critical: float = DEFAULT_CRITICAL) -> pd.DataFrame:
    preds, units = predict_last_windows(art, test_df)
    df = pd.DataFrame({"motor": units, "rul_estimada": preds})
    df["estado"] = [rul_status(r, warning, critical) for r in preds]
    return df.sort_values("rul_estimada").reset_index(drop=True)