"""Módulos 1 y 2 — Limpieza, normalización, etiquetas de RUL y ventanas temporales.

Decisiones metodológicas clave (ver README, sección "Decisiones y diferencias con el protocolo"):

- La normalización se ajusta SOLO con datos de entrenamiento y se aplica por *régimen de
  operación* (no por motor individual): en prueba los motores están truncados y no se
  conoce su trayectoria completa, por lo que un min-max por motor usaría información
  distinta en entrenamiento y en inferencia.
- La partición entrenamiento/validación se hace por MOTOR, nunca por ventana, para evitar
  fuga de información entre ventanas solapadas de un mismo motor.
- Se incluye ingeniería de características temporal (medias y desviaciones móviles) 
  para capturar la velocidad y tendencia de degradación de los sensores.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import OP_COLS, SENSOR_COLS

def add_rul(df: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """Añade ``rul`` (lineal, sin límite) y ``rul_target`` (lineal por tramos, con tope).

    RUL(t) = ciclo_final_del_motor − t. La función lineal por tramos (Heimes, 2008) asume
    que durante el funcionamiento sano la degradación es despreciable, por lo que se
    limita la RUL a ``cap`` ciclos.
    """
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = (max_cycle - out["cycle"]).astype(float)
    out["rul_target"] = out["rul"].clip(upper=cap) if cap is not None else out["rul"]
    return out

# Normalización por régimen de operación + Ingeniería de características temporal
class Preprocessor:
    def __init__(
        self, 
        n_regimes: int = 1, 
        std_tol: float = 0.01, 
        roll_window: int = 5, 
        seed: int = 42
    ):
        self.n_regimes = n_regimes
        self.std_tol = std_tol
        self.roll_window = roll_window
        self.seed = seed
        self.op_scaler_: StandardScaler | None = None
        self.kmeans_: KMeans | None = None
        self.base_feature_cols_: list[str] = []
        self.feature_cols_: list[str] = []
        self.dropped_cols_: list[str] = []
        self.mins_: np.ndarray | None = None
        self.ranges_: np.ndarray | None = None
        self.is_fitted_ = False

    # regímenes 
    def assign_regimes(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_regimes == 1:
            return np.zeros(len(df), dtype=int)
        ops = self.op_scaler_.transform(df[OP_COLS].to_numpy(dtype=float))
        return self.kmeans_.predict(ops)

    # ajuste 
    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        if self.n_regimes > 1:
            self.op_scaler_ = StandardScaler().fit(train_df[OP_COLS].to_numpy(dtype=float))
            self.kmeans_ = KMeans(n_clusters=self.n_regimes, n_init=10, random_state=self.seed)
            self.kmeans_.fit(self.op_scaler_.transform(train_df[OP_COLS].to_numpy(dtype=float)))
        regimes = self.assign_regimes(train_df)
        sensors = train_df[SENSOR_COLS].reset_index(drop=True)
        reg = pd.Series(regimes)

        within_std = sensors.groupby(reg).std(ddof=0).fillna(0.0).max(axis=0)
        self.base_feature_cols_ = [c for c in SENSOR_COLS if within_std[c] > self.std_tol]
        self.dropped_cols_ = [c for c in SENSOR_COLS if c not in self.base_feature_cols_]
        
        if not self.base_feature_cols_:
            raise ValueError("Ningún sensor supera el umbral de varianza; revise los datos.")

        # Expandir características para incluir medias y desviaciones móviles temporales
        rolling_features = []
        for c in self.base_feature_cols_:
            rolling_features.extend([f"{c}_roll_mean", f"{c}_roll_std"])
        self.feature_cols_ = self.base_feature_cols_ + rolling_features

        feats = sensors[self.base_feature_cols_]
        mins = feats.groupby(reg).min().reindex(range(self.n_regimes))
        maxs = feats.groupby(reg).max().reindex(range(self.n_regimes))
        ranges = np.array((maxs - mins).to_numpy(dtype=float), copy=True)  # copia editable
        ranges[ranges == 0] = 1.0               # evita división entre cero
        self.mins_ = np.array(mins.fillna(0.0).to_numpy(dtype=float), copy=True)
        self.ranges_ = np.nan_to_num(ranges, nan=1.0)
        self.is_fitted_ = True
        return self

    # transformación e ingeniería de características
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor no ajustado: llame a fit() primero.")
        
        regimes = self.assign_regimes(df)
        vals = df[self.base_feature_cols_].to_numpy(dtype=float)
        scaled = (vals - self.mins_[regimes]) / self.ranges_[regimes]

        keep = [c for c in ("unit", "cycle", "rul", "rul_target") if c in df.columns]
        out = df[keep].reset_index(drop=True).copy()
        
        # Asignar sensores base escalados
        out[self.base_feature_cols_] = scaled.astype(np.float32)
        out["regime"] = regimes

        # Aplicar características temporales (rolling mean y std) agrupadas por motor de forma causal
        roll_dfs = []
        for unit, g in out.groupby("unit", sort=True):
            g_sorted = g.sort_values("cycle").copy()
            for c in self.base_feature_cols_:
                g_sorted[f"{c}_roll_mean"] = g_sorted[c].rolling(self.roll_window, min_periods=1).mean()
                g_sorted[f"{c}_roll_std"] = g_sorted[c].rolling(self.roll_window, min_periods=1).std().fillna(0.0)
            roll_dfs.append(g_sorted)

        out = pd.concat(roll_dfs).sort_index().reset_index(drop=True)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # persistencia 
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "Preprocessor":
        return joblib.load(path)

# Ventanas temporales
def sliding_windows(values: np.ndarray, window: int, pad_start: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    n = len(values)
    if pad_start:
        pad = np.repeat(values[:1], window - 1, axis=0)
        values = np.vstack([pad, values])
    elif n < window:
        pad = np.repeat(values[:1], window - n, axis=0)
        values = np.vstack([pad, values])
    view = sliding_window_view(values, window_shape=(window, values.shape[1]))
    return np.ascontiguousarray(view[:, 0])


def build_training_windows(
    df: pd.DataFrame, feature_cols: list[str], window: int, target_col: str = "rul_target"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ventanas de entrenamiento/validación. La etiqueta es la RUL del último ciclo de la
    ventana. Devuelve ``X (N, W, F)``, ``y (N,)`` y ``units (N,)``."""
    xs, ys, us = [], [], []
    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        w = sliding_windows(g[feature_cols].to_numpy(), window, pad_start=False)
        y = g[target_col].to_numpy(dtype=np.float32)[-len(w):]
        xs.append(w)
        ys.append(y)
        us.append(np.full(len(w), unit, dtype=int))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(us)


def build_test_windows(
    df: pd.DataFrame, feature_cols: list[str], window: int
) -> tuple[np.ndarray, np.ndarray]:
    # Una ventana por motor de prueba: los últimos ``window`` ciclos observados
    xs, units = [], []
    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        w = sliding_windows(g[feature_cols].to_numpy(), window, pad_start=False)
        xs.append(w[-1])
        units.append(unit)
    return np.stack(xs), np.asarray(units, dtype=int)

# Partición por motor
def split_by_engine(
    units, val_fraction: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Divide los identificadores de motor en entrenamiento (1−f) y validación (f)."""
    units = np.unique(np.asarray(units))
    if len(units) < 2:
        raise ValueError("Se necesitan al menos 2 motores para particionar.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(units)
    n_val = int(np.clip(round(len(units) * val_fraction), 1, len(units) - 1))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])