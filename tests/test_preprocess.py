"""Tests for src/preprocess.py — normalization and cloud mask logic"""
import numpy as np
import pytest
from src.preprocess import normalize_reflectance, apply_cloud_mask


class TestNormalizeReflectance:
    def test_output_range_zero_to_one(self):
        image = np.random.uniform(0.0, 0.5, (5, 100, 100)).astype(np.float32)
        result = normalize_reflectance(image, clip_percentile=98)
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_shape_preserved(self):
        image = np.ones((3, 64, 64), dtype=np.float32) * 0.3
        result = normalize_reflectance(image)
        assert result.shape == (3, 64, 64)

    def test_constant_band_handled(self):
        """A completely flat band should not cause division errors"""
        image = np.ones((2, 10, 10), dtype=np.float32) * 0.3
        result = normalize_reflectance(image)
        assert not np.any(np.isnan(result)), "NaN found in normalized constant band"
        assert not np.any(np.isinf(result))

    def test_all_nan_band_skipped(self):
        """An all-NaN band should be preserved as zeros, not crash"""
        image = np.full((2, 10, 10), np.nan, dtype=np.float32)
        result = normalize_reflectance(image)  # should not raise
        assert result is not None


class TestApplyCloudMask:
    def test_scl_masks_cloud_pixels(self):
        image  = np.ones((3, 5, 5), dtype=np.float32)
        # SCL class 9 = high probability cloud
        scl    = np.zeros((5, 5), dtype=np.uint8)
        scl[2, 2] = 9
        result = apply_cloud_mask(image, scl_band=scl)
        assert np.isnan(result[:, 2, 2]).all(), "Cloud pixel should be NaN"
        assert not np.isnan(result[:, 0, 0]).any(), "Clear pixel should not be NaN"

    def test_custom_mask_applied(self):
        image  = np.ones((2, 4, 4), dtype=np.float32) * 0.5
        mask   = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        result = apply_cloud_mask(image, custom_mask=mask)
        assert np.isnan(result[:, 0, 0]).all()
        assert not np.isnan(result[:, 3, 3]).any()

    def test_no_mask_returns_unchanged(self):
        image  = np.ones((2, 5, 5), dtype=np.float32)
        result = apply_cloud_mask(image)
        np.testing.assert_array_equal(result, image)

    def test_output_shape_preserved(self):
        image  = np.random.rand(4, 20, 30).astype(np.float32)
        mask   = np.zeros((20, 30), dtype=bool)
        result = apply_cloud_mask(image, custom_mask=mask)
        assert result.shape == (4, 20, 30)
