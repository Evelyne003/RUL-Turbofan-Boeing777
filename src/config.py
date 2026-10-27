from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = Path(os.environ.get("RUL_DATA_DIR", ROOT / "data" / "raw"))
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = Path(os.environ.get("RUL_MODELS_DIR", ROOT / "models"))

SUBSETS = ("FD001", "FD002", "FD003", "FD004")

N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}

OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *OP_COLS, *SENSOR_COLS]

# Descripción de los 21 canales (Frederick et al., 2007; Saxena et al., 2008).
SENSOR_DESCRIPTIONS = {
    "s1": "T2 – Temperatura total a la entrada del fan",
    "s2": "T24 – Temperatura total a la salida del compresor de baja (LPC)",
    "s3": "T30 – Temperatura total a la salida del compresor de alta (HPC)",
    "s4": "T50 – Temperatura total a la salida de la turbina de baja (LPT)",
    "s5": "P2 – Presión a la entrada del fan",
    "s6": "P15 – Presión total en el ducto de bypass",
    "s7": "P30 – Presión total a la salida del HPC",
    "s8": "Nf – Velocidad física del fan",
    "s9": "Nc – Velocidad física del núcleo (compresor)",
    "s10": "epr – Razón de presión del motor (P50/P2)",
    "s11": "Ps30 – Presión estática a la salida del HPC",
    "s12": "phi – Razón flujo de combustible / Ps30",
    "s13": "NRf – Velocidad corregida del fan",
    "s14": "NRc – Velocidad corregida del núcleo",
    "s15": "BPR – Razón de bypass",
    "s16": "farB – Razón combustible/aire del quemador",
    "s17": "htBleed – Entalpía de sangrado",
    "s18": "Nf_dmd – Velocidad del fan demandada",
    "s19": "PCNfR_dmd – Velocidad corregida del fan demandada",
    "s20": "W31 – Sangrado de enfriamiento de la HPT",
    "s21": "W32 – Sangrado de enfriamiento de la LPT",
}

LITERATURE_SENSORS_FD001 = [
    "s2", "s3", "s4", "s7", "s8", "s9", "s11",
    "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]


@dataclass
class TrainConfig:
    """Hiperparámetros óptimos del experimento para la arquitectura LSTM."""

    subset: str = "FD001"
    window: int = 60                    # Ventana temporal óptima de 60 ciclos
    rul_cap: int = 125                  # Umbral de la RUL lineal por tramos
    val_fraction: float = 0.20          # 20 % de motores para validación interna
    seed: int = 42

    lstm_units: tuple = (64, 64)        # Arquitectura apilada óptima (64, 64)
    dense_units: tuple = (32, 16)       # Capas densas posteriores
    dropout: float = 0.1                # Dropout de regularización

    learning_rate: float = 1e-3         # Adam 
    lr_factor: float = 0.4              # Factor óptimo de reducción ReduceLROnPlateau
    lr_patience: int = 5
    min_lr: float = 1e-5
    early_stop_patience: int = 15
    epochs: int = 100
    batch_size: int = 256

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lstm_units"] = list(self.lstm_units)
        d["dense_units"] = list(self.dense_units)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = dict(d)
        d["lstm_units"] = tuple(d.get("lstm_units", cls.lstm_units))
        d["dense_units"] = tuple(d.get("dense_units", cls.dense_units))
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)