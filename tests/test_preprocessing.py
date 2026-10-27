import numpy as np
import pytest

from src.preprocessing import (Preprocessor, add_rul, build_test_windows,
                               build_training_windows, sliding_windows, split_by_engine)
from tests.synthetic import make_engines


def test_add_rul_linear_and_capped():
    df, _ = make_engines(3, seed=1)
    out = add_rul(df, cap=50)
    last = out.groupby("unit").tail(1)
    assert (last["rul"] == 0).all()
    assert out["rul_target"].max() <= 50
    first = out.groupby("unit").head(1)
    assert (first["rul"] == out.groupby("unit")["cycle"].max().to_numpy() - 1).all()


def test_constant_sensors_dropped_single_regime():
    df, _ = make_engines(10, n_regimes=1, seed=2)
    pre = Preprocessor(n_regimes=1).fit(df)
    for s in ("s1", "s5", "s10", "s16", "s18", "s19"):
        assert s in pre.dropped_cols_
    assert len(pre.feature_cols_) == 15


def test_transform_range_on_train_is_unit_interval():
    df, _ = make_engines(10, n_regimes=1, seed=3)
    pre = Preprocessor(n_regimes=1).fit(df)
    out = pre.transform(df)
    vals = out[pre.feature_cols_].to_numpy()
    assert vals.min() >= -1e-6 and vals.max() <= 1 + 1e-6


def test_multi_regime_clusters_and_normalization():
    df, _ = make_engines(15, n_regimes=6, seed=4)
    pre = Preprocessor(n_regimes=6).fit(df)
    reg = pre.assign_regimes(df)
    assert set(np.unique(reg)) == set(range(6))
    out = pre.transform(df)
    assert out[pre.feature_cols_].to_numpy().max() <= 1 + 1e-6
    assert "s1" in pre.dropped_cols_


def test_preprocessor_not_fitted_raises():
    df, _ = make_engines(2)
    with pytest.raises(RuntimeError):
        Preprocessor().transform(df)


def test_preprocessor_save_load(tmp_path):
    df, _ = make_engines(5, seed=5)
    pre = Preprocessor().fit(df)
    pre.save(tmp_path / "p.joblib")
    pre2 = Preprocessor.load(tmp_path / "p.joblib")
    a = pre.transform(df)[pre.feature_cols_].to_numpy()
    b = pre2.transform(df)[pre2.feature_cols_].to_numpy()
    np.testing.assert_allclose(a, b)


def test_sliding_windows_shapes_and_padding():
    v = np.arange(20, dtype=float).reshape(10, 2)
    w = sliding_windows(v, 4)
    assert w.shape == (7, 4, 2)
    np.testing.assert_array_equal(w[0], v[:4])
    np.testing.assert_array_equal(w[-1], v[-4:])
    short = sliding_windows(v[:3], 5)
    assert short.shape == (1, 5, 2)
    np.testing.assert_array_equal(short[0, 0], v[0])      # relleno replicando la primera lectura
    padded = sliding_windows(v, 4, pad_start=True)
    assert padded.shape == (10, 4, 2)
    np.testing.assert_array_equal(padded[-1], v[-4:])


def test_training_windows_labels_align_with_last_cycle():
    df, _ = make_engines(4, seed=6)
    df = add_rul(df, cap=125)
    pre = Preprocessor().fit(df)
    n = pre.transform(df)
    X, y, units = build_training_windows(n, pre.feature_cols_, window=10)
    for u in np.unique(units):
        g = n[n["unit"] == u]
        assert (units == u).sum() == len(g) - 9
        assert y[units == u][-1] == g["rul_target"].iloc[-1] == 0


def test_test_windows_one_per_engine():
    df, _ = make_engines(6, seed=7, truncated=True)
    pre = Preprocessor().fit(df)
    X, units = build_test_windows(pre.transform(df), pre.feature_cols_, window=25)
    assert X.shape[0] == 6 and X.shape[1] == 25
    assert list(units) == sorted(df["unit"].unique())


def test_split_by_engine_disjoint_and_reproducible():
    units = np.arange(1, 101)
    tr, va = split_by_engine(units, 0.2, seed=1)
    assert len(va) == 20 and len(tr) == 80
    assert set(tr).isdisjoint(va)
    tr2, va2 = split_by_engine(units, 0.2, seed=1)
    np.testing.assert_array_equal(va, va2)


def test_scaler_fit_uses_only_given_data():
    """Verifica que los parámetros dependen exclusivamente de los datos de ajuste."""
    df, _ = make_engines(10, seed=8)
    fit_df = df[df["unit"] <= 7]
    pre = Preprocessor().fit(fit_df)
    expected_min = fit_df[pre.feature_cols_].min().to_numpy()
    np.testing.assert_allclose(pre.mins_[0], expected_min)
