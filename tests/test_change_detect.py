"""Tests for src/change_detect.py — differencing and mask logic"""
import numpy as np
import pytest
from src.change_detect import compute_difference, identify_deforestation, identify_urbanization


class TestComputeDifference:
    def test_basic_subtraction(self):
        t1 = np.array([[0.3, 0.5]], dtype=np.float32)
        t2 = np.array([[0.1, 0.8]], dtype=np.float32)
        result = compute_difference(t1, t2)
        np.testing.assert_allclose(result, [[-0.2, 0.3]], atol=1e-6)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same spatial dimensions"):
            compute_difference(np.zeros((5, 5)), np.zeros((4, 5)))

    def test_no_change_returns_zeros(self):
        arr = np.random.rand(20, 20).astype(np.float32)
        result = compute_difference(arr, arr)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)


class TestIdentifyDeforestation:
    def test_large_ndvi_drop_detected(self):
        t1 = np.full((10, 10), 0.7, dtype=np.float32)
        t2 = np.full((10, 10), 0.2, dtype=np.float32)  # drop = 0.5
        mask = identify_deforestation(t1, t2, threshold=0.2)
        assert mask.all(), "All pixels should be flagged as deforested"

    def test_small_ndvi_drop_not_flagged(self):
        t1 = np.full((10, 10), 0.5, dtype=np.float32)
        t2 = np.full((10, 10), 0.45, dtype=np.float32)  # drop = 0.05
        mask = identify_deforestation(t1, t2, threshold=0.2)
        assert not mask.any(), "Small drop should not be flagged"

    def test_output_is_binary(self):
        t1 = np.random.rand(20, 20).astype(np.float32)
        t2 = np.random.rand(20, 20).astype(np.float32)
        mask = identify_deforestation(t1, t2)
        unique_vals = np.unique(mask)
        assert set(unique_vals).issubset({0, 1}), "Mask must be binary (0 or 1)"

    def test_output_dtype(self):
        t1 = np.ones((5, 5), dtype=np.float32)
        t2 = np.zeros((5, 5), dtype=np.float32)
        mask = identify_deforestation(t1, t2)
        assert mask.dtype == np.uint8


class TestIdentifyUrbanization:
    def test_large_ndbi_gain_detected(self):
        t1 = np.full((10, 10), 0.0,  dtype=np.float32)
        t2 = np.full((10, 10), 0.5,  dtype=np.float32)  # gain = 0.5
        mask = identify_urbanization(t1, t2, threshold=0.15)
        assert mask.all()

    def test_no_gain_not_flagged(self):
        t1 = np.full((10, 10), 0.3, dtype=np.float32)
        t2 = np.full((10, 10), 0.3, dtype=np.float32)
        mask = identify_urbanization(t1, t2)
        assert not mask.any()
