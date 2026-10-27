import numpy as np
import pytest

from src.config import COLUMNS
from src.ingestion import load_subset, missing_files
from tests.synthetic import write_subset


def test_load_subset_roundtrip(tmp_path):
    write_subset(tmp_path, "FD001", n_train=5, n_test=3)
    d = load_subset("FD001", tmp_path)
    assert list(d.train.columns) == COLUMNS
    assert d.train["unit"].nunique() == 5
    assert d.test["unit"].nunique() == 3
    assert len(d.rul_test) == 3
    assert np.all(d.rul_test >= 0)


def test_missing_files_detected(tmp_path):
    assert "train_FD001.txt" in missing_files(tmp_path, subsets=("FD001",))
    with pytest.raises(FileNotFoundError):
        load_subset("FD001", tmp_path)


def test_invalid_subset(tmp_path):
    with pytest.raises(ValueError):
        load_subset("FD999", tmp_path)


def test_rul_length_mismatch(tmp_path):
    write_subset(tmp_path, "FD001", n_train=5, n_test=3)
    (tmp_path / "RUL_FD001.txt").write_text("10\n20\n")
    with pytest.raises(ValueError):
        load_subset("FD001", tmp_path)