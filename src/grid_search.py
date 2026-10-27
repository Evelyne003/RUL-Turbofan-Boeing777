from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_RAW, MODELS_DIR, SUBSETS, TrainConfig
from src.train import prepare_data, train_model

logger = logging.getLogger(__name__)


def run_grid(base: TrainConfig, windows, lstm_options, dropouts, raw_dir: Path | None = None) -> pd.DataFrame:
    rows = []
    for window in windows:
        cfg_w = replace(base, window=window)
        bundle = prepare_data(cfg_w, raw_dir)          # los datos dependen solo de la ventana
        for units, drop in itertools.product(lstm_options, dropouts):
            cfg = replace(cfg_w, lstm_units=tuple(units), dropout=float(drop))
            _, hist = train_model(cfg, bundle)
            best = int(hist["val_loss"].idxmin())
            rows.append({
                "subset": cfg.subset, "window": window, "lstm_units": tuple(units), "dropout": drop,
                "val_rmse": float(np.sqrt(hist.loc[best, "val_loss"])),
                "best_epoch": best + 1, "epochs_run": len(hist),
            })
            logger.info("%s", rows[-1])
    return pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Búsqueda en cuadrícula (validación interna)")
    p.add_argument("--subset", default="FD001", choices=SUBSETS)
    p.add_argument("--windows", type=int, nargs="+", default=[15, 20, 30, 40])
    p.add_argument("--lstm-options", type=str, nargs="+", default=["32,32", "64,32", "64,64", "128,64"],
                   help="p. ej. 64,32 128,64 (celdas de cada capa LSTM separadas por coma)")
    p.add_argument("--dropouts", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--raw-dir", default=str(DATA_RAW))
    p.add_argument("--out", default=str(MODELS_DIR / "grid_search"))
    a = p.parse_args(argv)

    options = [tuple(int(x) for x in s.split(",")) for s in a.lstm_options]
    base = TrainConfig(subset=a.subset, epochs=a.epochs)
    df = run_grid(base, a.windows, options, a.dropouts, Path(a.raw_dir))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"grid_{a.subset}.csv"
    df.to_csv(path, index=False)
    print(df.head(10).to_string(index=False))
    print(f"\nMejor configuración (por val_rmse) guardada en {path}")


if __name__ == "__main__":
    main()