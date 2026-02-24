"""Tests for src/report.py — area statistics calculation"""
import numpy as np
import pytest
from src.report import generate_area_stats, generate_class_area_report


class TestGenerateAreaStats:
    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        stats = generate_area_stats(mask, pixel_resolution_m=10.0)
        assert stats["pixels"] == 0
        assert stats["area_sq_km"] == 0.0

    def test_full_mask(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        stats = generate_area_stats(mask, pixel_resolution_m=10.0)
        # 10000 pixels × (10m)² = 1,000,000 m² = 1.0 km²
        assert stats["pixels"] == 10000
        assert abs(stats["area_sq_km"] - 1.0) < 1e-6

    def test_partial_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:50, :50] = 1   # 2500 pixels = 0.25 km² at 10m
        stats = generate_area_stats(mask, pixel_resolution_m=10.0)
        assert stats["pixels"] == 2500
        assert abs(stats["area_sq_km"] - 0.25) < 1e-6

    def test_different_resolution(self):
        """At 30m resolution, one pixel = 900 m²"""
        mask = np.ones((10, 10), dtype=np.uint8)   # 100 pixels
        stats = generate_area_stats(mask, pixel_resolution_m=30.0)
        # 100 × 900 m² = 90000 m² = 0.09 km²
        assert abs(stats["area_sq_km"] - 0.09) < 1e-6


class TestGenerateClassAreaReport:
    def test_all_classes_present(self):
        class_map = np.zeros((10, 10), dtype=np.uint8)
        class_map[5:, :] = 1   # Vegetation bottom half
        report = generate_class_area_report(class_map, pixel_resolution_m=10.0)
        assert "Vegetation" in report
        assert "Urban" in report
        assert "Water" in report

    def test_percentages_sum_to_100(self):
        class_map = np.zeros((10, 10), dtype=np.uint8)
        class_map[5:, :] = 1
        report = generate_class_area_report(class_map)
        total_pct = sum(v["percentage"] for v in report.values())
        assert abs(total_pct - 100.0) < 0.1
