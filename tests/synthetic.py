from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import COLUMNS

_REGIME_TABLE = np.array([
    [0.0, 0.0, 100.0], [10.0, 0.25, 100.0], [20.0, 0.70, 100.0],
    [25.0, 0.62, 60.0], [35.0, 0.84, 100.0], [42.0, 0.84, 100.0],
])
_CONSTANT_SENSORS = {1, 5, 10, 16, 18, 19}


def make_engines(n_units: int = 20, n_regimes: int = 1, seed: int = 0, truncated: bool = False):
    rng = np.random.default_rng(seed)
    base = 100 + 10 * np.arange(1, 22)
    slopes = rng.uniform(-6, 6, size=21)
    regime_offsets = rng.uniform(5, 40, size=(6, 21))
    rows, rul_final = [], []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(130, 220))
        T = int(rng.integers(30, life - 10)) if truncated else life
        t = np.arange(1, T + 1)
        reg = rng.integers(0, n_regimes, size=T) if n_regimes > 1 else np.zeros(T, dtype=int)
        ops = _REGIME_TABLE[reg] + rng.normal(0, 1e-3, size=(T, 3))
        health = (t / life) ** 2
        data = np.empty((T, 21))
        for i in range(21):
            if (i + 1) in _CONSTANT_SENSORS:
                data[:, i] = base[i]
            else:
                data[:, i] = (base[i] + regime_offsets[reg, i] * (n_regimes > 1)
                              + slopes[i] * health * (1 + 0.1 * rng.normal())
                              + rng.normal(0, 0.3, size=T))
        block = np.column_stack([np.full(T, unit), t, ops, data])
        rows.append(block)
        rul_final.append(life - T)
    df = pd.DataFrame(np.vstack(rows), columns=COLUMNS)
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df, np.asarray(rul_final, dtype=float)


def write_subset(raw_dir: Path, subset: str = "FD001", n_regimes: int = 1,
                 n_train: int = 20, n_test: int = 8, seed: int = 0) -> None:
    """Escribe train_/test_/RUL_ en el formato texto original (separado por espacios)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    train, _ = make_engines(n_train, n_regimes, seed)
    test, rul = make_engines(n_test, n_regimes, seed + 1, truncated=True)
    train.to_csv(raw_dir / f"train_{subset}.txt", sep=" ", header=False, index=False)
    test.to_csv(raw_dir / f"test_{subset}.txt", sep=" ", header=False, index=False)
    np.savetxt(raw_dir / f"RUL_{subset}.txt", rul, fmt="%d")