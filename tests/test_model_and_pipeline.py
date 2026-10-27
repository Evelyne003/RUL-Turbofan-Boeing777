import json

import numpy as np

from src.config import TrainConfig
from src.inference import (fleet_status, load_artifacts, predict_engine_curve,
                           predict_last_windows, rul_status, true_rul_curve)
from src.ingestion import load_subset
from src.model import build_lstm_model
from src.train import run
from tests.synthetic import write_subset


def test_model_output_shape_and_compile():
    m = build_lstm_model(window=12, n_features=7, lstm_units=(8, 4), dense_units=(6,), dropout=0.1)
    out = m.predict(np.zeros((5, 12, 7), dtype="float32"), verbose=0)
    assert out.shape == (5, 1)
    assert abs(float(m.optimizer.learning_rate) - 1e-3) < 1e-9


def test_rul_status_thresholds():
    assert rul_status(10, 60, 30) == "CRÍTICO"
    assert rul_status(45, 60, 30) == "PRECAUCIÓN"
    assert rul_status(100, 60, 30) == "NORMAL"


def test_true_rul_curve():
    c = true_rul_curve(5, rul_final=10)
    assert c[-1] == 10 and c[0] == 14
    assert true_rul_curve(5, 200, cap=125).max() == 125


def test_end_to_end_smoke(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "models"
    write_subset(raw, "FD001", n_train=14, n_test=6)
    cfg = TrainConfig(subset="FD001", window=10, epochs=2, batch_size=64,
                      lstm_units=(8,), dense_units=(8,))
    metrics = run(cfg, raw, out)
    for key in ("rmse", "mae", "score"):
        assert np.isfinite(metrics["test"][key])
    d = out / "FD001"
    for f in ("model.keras", "preprocessor.joblib", "config.json", "metrics.json",
              "history.csv", "test_predictions.csv", "figures/pred_vs_real.png"):
        assert (d / f).exists(), f
    assert json.loads((d / "config.json").read_text())["window"] == 10

    art = load_artifacts("FD001", out)
    data = load_subset("FD001", raw)
    preds, units = predict_last_windows(art, data.test)
    assert len(preds) == len(units) == 6
    assert preds.min() >= 0 and preds.max() <= cfg.rul_cap
    eng = data.test[data.test["unit"] == units[0]]
    curve = predict_engine_curve(art, eng)
    assert len(curve) == len(eng)
    fleet = fleet_status(art, data.test)
    assert list(fleet["rul_estimada"]) == sorted(fleet["rul_estimada"])