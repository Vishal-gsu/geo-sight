"""Tests for src/spectral.py — NDVI, NDWI, NDBI math"""
import numpy as np
import pytest
from src.spectral import calculate_ndvi, calculate_ndwi, calculate_ndbi

# ─── Fixtures ─────────────────────────────────────────────
@pytest.fixture
def uniform_high_nir():
    """NIR >> Red → should produce high NDVI (healthy vegetation)"""
    nir = np.full((10, 10), 0.8, dtype=np.float32)
    red = np.full((10, 10), 0.1, dtype=np.float32)
    return nir, red

@pytest.fixture
def uniform_equal():
    """NIR == Red → NDVI should be ~0"""
    arr = np.full((10, 10), 0.5, dtype=np.float32)
    return arr, arr

# ─── NDVI Tests ───────────────────────────────────────────
class TestNDVI:
    def test_high_vegetation(self, uniform_high_nir):
        nir, red = uniform_high_nir
        result = calculate_ndvi(nir, red)
        assert result.mean() > 0.5, "High NIR should yield NDVI > 0.5"

    def test_equal_bands_near_zero(self, uniform_equal):
        arr, _ = uniform_equal
        result = calculate_ndvi(arr, arr)
        assert np.allclose(result, 0.0, atol=1e-5), "Equal NIR/Red should yield NDVI ≈ 0"

    def test_output_range_clipped(self):
        """NDVI must always be in [-1, 1]"""
        nir = np.random.rand(100, 100).astype(np.float32)
        red = np.random.rand(100, 100).astype(np.float32)
        result = calculate_ndvi(nir, red)
        assert result.min() >= -1.0 and result.max() <= 1.0

    def test_output_shape_preserved(self):
        arr = np.ones((50, 80), dtype=np.float32)
        result = calculate_ndvi(arr * 0.7, arr * 0.3)
        assert result.shape == (50, 80)

    def test_no_division_by_zero(self):
        """All-zero inputs must not crash"""
        zeros = np.zeros((10, 10), dtype=np.float32)
        result = calculate_ndvi(zeros, zeros)
        assert not np.any(np.isnan(result)), "NaN found in NDVI with zero inputs"
        assert not np.any(np.isinf(result)), "Inf found in NDVI with zero inputs"

    def test_water_pixels_negative(self):
        """Water: NIR ≈ 0, Green/Red moderate → NDVI should be negative"""
        nir = np.full((5, 5), 0.02, dtype=np.float32)
        red = np.full((5, 5), 0.05, dtype=np.float32)
        result = calculate_ndvi(nir, red)
        assert result.mean() < 0.0, "Water-like pixels should have negative NDVI"


# ─── NDWI Tests ──────────────────────────────────────────
class TestNDWI:
    def test_water_positive(self):
        green = np.full((5, 5), 0.5, dtype=np.float32)
        nir   = np.full((5, 5), 0.1, dtype=np.float32)
        result = calculate_ndwi(green, nir)
        assert result.mean() > 0.0, "High Green/Low NIR should yield positive NDWI (water)"

    def test_vegetation_negative(self):
        green = np.full((5, 5), 0.1, dtype=np.float32)
        nir   = np.full((5, 5), 0.7, dtype=np.float32)
        result = calculate_ndwi(green, nir)
        assert result.mean() < 0.0, "High NIR should yield negative NDWI (not water)"

    def test_output_range(self):
        g = np.random.rand(50, 50).astype(np.float32)
        n = np.random.rand(50, 50).astype(np.float32)
        result = calculate_ndwi(g, n)
        assert result.min() >= -1.0 and result.max() <= 1.0

    def test_no_nan_on_zeros(self):
        result = calculate_ndwi(np.zeros((5,5)), np.zeros((5,5)))
        assert not np.any(np.isnan(result))


# ─── NDBI Tests ──────────────────────────────────────────
class TestNDBI:
    def test_urban_positive(self):
        swir = np.full((5, 5), 0.5, dtype=np.float32)
        nir  = np.full((5, 5), 0.2, dtype=np.float32)
        result = calculate_ndbi(swir, nir)
        assert result.mean() > 0.0, "High SWIR/Low NIR should yield positive NDBI (urban)"

    def test_vegetation_negative(self):
        swir = np.full((5, 5), 0.1, dtype=np.float32)
        nir  = np.full((5, 5), 0.7, dtype=np.float32)
        result = calculate_ndbi(swir, nir)
        assert result.mean() < 0.0

    def test_output_range(self):
        s = np.random.rand(50, 50).astype(np.float32)
        n = np.random.rand(50, 50).astype(np.float32)
        result = calculate_ndbi(s, n)
        assert result.min() >= -1.0 and result.max() <= 1.0
