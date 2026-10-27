import numpy as np

from src.evaluation import mae, nasa_score, regression_report, rmse


def test_perfect_prediction():
    y = np.array([10.0, 50.0, 100.0])
    assert rmse(y, y) == 0 and mae(y, y) == 0 and nasa_score(y, y) == 0


def test_rmse_mae_values():
    y, p = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    assert np.isclose(rmse(y, p), np.sqrt((9 + 16) / 2))
    assert np.isclose(mae(y, p), 3.5)


def test_score_formula_exact():
    # d = pred - true. d=-10 -> exp(10/13)-1 ; d=+10 -> exp(10/10)-1
    assert np.isclose(nasa_score([50.0], [40.0]), np.exp(10 / 13) - 1)
    assert np.isclose(nasa_score([50.0], [60.0]), np.exp(10 / 10) - 1)


def test_score_penalizes_late_prediction_more():
    """Sobreestimar la RUL (estimación tardía) debe costar más que subestimarla."""
    assert nasa_score([50.0], [60.0]) > nasa_score([50.0], [40.0])


def test_report_keys():
    r = regression_report([10, 20, 30], [12, 18, 35])
    assert set(r) == {"rmse", "mae", "score", "n", "pct_estimacion_tardia"}
    assert r["n"] == 3
