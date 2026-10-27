"""Módulo 1 — Adquisición de datos C-MAPSS

Formato de los archivos (Saxena et al., 2008): texto separado por espacios, 26 columnas
(unidad, ciclo, 3 ajustes operacionales, 21 sensores). Para cada subconjunto FD00X:
``train_FD00X.txt`` (trayectorias completas hasta la falla), ``test_FD00X.txt``
(trayectorias truncadas antes de la falla) y ``RUL_FD00X.txt`` (RUL verdadera del último
ciclo observado de cada motor de prueba).
"""
from __future__ import annotations

import argparse
import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import COLUMNS, DATA_RAW, SUBSETS

logger = logging.getLogger(__name__)

CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

@dataclass
class CMAPSSData:
    """Contenedor de un subconjunto C-MAPSS."""
    subset: str
    train: pd.DataFrame
    test: pd.DataFrame
    rul_test: np.ndarray  


def expected_files(subset: str) -> list[str]:
    return [f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"]


def missing_files(raw_dir: Path = DATA_RAW, subsets=SUBSETS) -> list[str]:
    raw_dir = Path(raw_dir)
    return [f for s in subsets for f in expected_files(s) if not (raw_dir / f).exists()]


def _extract_zip(data: bytes, dest: Path) -> None:
    """Extrae los .txt de forma plana; soporta zips anidados."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if not name:
                continue
            if name.lower().endswith(".zip"):
                _extract_zip(zf.read(member), dest)
            elif name.lower().endswith(".txt"):
                (dest / name).write_bytes(zf.read(member))


def download_cmapss(dest: Path = DATA_RAW, url: str = CMAPSS_URL, timeout: int = 120) -> None:
    """Descarga y extrae C-MAPSS en ``dest``. Lanza RuntimeError si faltan archivos."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Descargando %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (URL de confianza)
        payload = resp.read()
    _extract_zip(payload, dest)
    missing = missing_files(dest)
    if missing:
        raise RuntimeError(f"Descarga incompleta; faltan: {missing}")
    logger.info("C-MAPSS disponible en %s", dest)


def read_cmapss_table(path: Path) -> pd.DataFrame:
    """Lee un archivo train_/test_FD00X.txt y valida su estructura."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name}: se esperaban {len(COLUMNS)} columnas y hay {df.shape[1]}")
    df.columns = COLUMNS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    if df.isna().any().any():
        raise ValueError(f"{path.name}: contiene valores faltantes")
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def read_rul_file(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None, names=["rul"])["rul"].to_numpy(dtype=float)


def load_subset(subset: str, raw_dir: Path | None = None) -> CMAPSSData:
    """Carga un subconjunto (FD001..FD004) y comprueba la consistencia test/RUL."""
    if subset not in SUBSETS:
        raise ValueError(f"Subconjunto inválido '{subset}'. Opciones: {SUBSETS}")
    raw_dir = Path(raw_dir) if raw_dir is not None else DATA_RAW
    missing = [f for f in expected_files(subset) if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan {missing} en {raw_dir}. Ejecute `make data` o copie los archivos manualmente."
        )
    train = read_cmapss_table(raw_dir / f"train_{subset}.txt")
    test = read_cmapss_table(raw_dir / f"test_{subset}.txt")
    rul = read_rul_file(raw_dir / f"RUL_{subset}.txt")
    if len(rul) != test["unit"].nunique():
        raise ValueError(
            f"{subset}: RUL_{subset}.txt tiene {len(rul)} filas pero test tiene "
            f"{test['unit'].nunique()} motores"
        )
    return CMAPSSData(subset=subset, train=train, test=test, rul_test=rul)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adquisición de C-MAPSS")
    parser.add_argument("--download", action="store_true", help="descarga y extrae C-MAPSS")
    parser.add_argument("--url", default=CMAPSS_URL)
    parser.add_argument("--raw-dir", default=str(DATA_RAW))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.download:
        download_cmapss(Path(args.raw_dir), args.url)
    missing = missing_files(Path(args.raw_dir))
    if missing:
        print("Archivos faltantes:", missing)
        raise SystemExit(1)
    for s in SUBSETS:
        d = load_subset(s, Path(args.raw_dir))
        print(f"{s}: train={d.train['unit'].nunique()} motores, "
              f"test={d.test['unit'].nunique()} motores, filas train={len(d.train)}")


if __name__ == "__main__":
    main()