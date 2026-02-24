"""
run_all.py — GeoSight One-Click Pipeline Runner

Run this single script to execute the ENTIRE GeoSight pipeline:
  1. Synthetic demo (instantly verifies all modules work)
  2. Real Sentinel-2 change detection (Bengaluru 2019 vs 2024)
  3. Download real satellite tile from GEE
  4. SAM automatic land cover segmentation

Usage:
    python run_all.py

Outputs saved to: results/
"""

import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    {
        "name": "Synthetic Demo",
        "script": "main.py",
        "desc": "Tests all modules with mock data — verifies your environment works",
        "critical": True,
    },
    {
        "name": "Real Change Detection (2019 → 2024)",
        "script": "bangalore_change_detection.py",
        "desc": "Pulls real Sentinel-2 data from GEE and analyses NDVI/NDBI changes over Bengaluru",
        "critical": True,
    },
    {
        "name": "Download Satellite Tile",
        "script": "download_tile.py",
        "desc": "Downloads a 2048×2048 real Bengaluru RGB patch (March 2024 scene)",
        "critical": True,
    },
    {
        "name": "SAM Segmentation",
        "script": "sam2_segmentation.py",
        "desc": "Runs Meta SAM (vit_b) zero-shot segmentation on the downloaded tile",
        "critical": False,  # SAM can fail on low-memory machines, that's OK
    },
]

WIDTH = 62

def banner():
    print("\n" + "═" * WIDTH)
    print("  🛰️  GeoSight — One-Click Pipeline Runner")
    print("  Sentinel-2 × SAM × Google Earth Engine")
    print("═" * WIDTH)
    print(f"  Python: {sys.version.split()[0]}  |  Output: results/")
    print("═" * WIDTH + "\n")

def run_step(step_num, step):
    label = f"[{step_num}/{len(STEPS)}] {step['name']}"
    print(f"\n{'─'*WIDTH}")
    print(f"  ▶  {label}")
    print(f"     {step['desc']}")
    print(f"{'─'*WIDTH}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, step["script"]],
        capture_output=True, text=True
    )
    elapsed = time.time() - t0

    # Print all output lines
    for line in result.stdout.splitlines():
        print(f"  {line}")
    if result.returncode != 0:
        print(f"\n  ⚠️  STDERR:")
        for line in result.stderr.splitlines()[-10:]:  # Last 10 lines of error
            print(f"     {line}")
        if step["critical"]:
            print(f"\n  ❌  Critical step failed. Stopping pipeline.")
            print(f"     Fix the error above, then re-run: python run_all.py")
            return False
        else:
            print(f"\n  ⚠️  Step failed but is non-critical — continuing...")
    else:
        print(f"\n  ✅  Done in {elapsed:.1f}s")
    return True


def results_summary():
    results_dir = Path("results")
    print(f"\n{'═'*WIDTH}")
    print("  📁  Results Summary — Open these files:")
    print(f"{'═'*WIDTH}")
    descriptions = {
        "demo_output.png":                 "Synthetic NDVI change detection demo",
        "bangalore_change_detection.png":  "6-panel NDVI/NDBI change charts (2019 vs 2024)",
        "bangalore_interactive_map.html":  "🗺️  Interactive map — OPEN IN YOUR BROWSER",
        "bangalore_rgb_sam2_overlay.png":  "SAM-detected segments on satellite image",
        "bangalore_class_overlay.png":     "Land cover classes overlaid on satellite",
        "bangalore_classification_map.png":"Clean categorical land cover map",
        "bangalore_sam2_summary.png":      "3-panel summary: RGB → SAM → Classification",
        "bangalore_class_stats.json":      "Area statistics in km² per land cover class",
    }
    for fname, desc in descriptions.items():
        fpath = results_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            size_str = f"{size/1024:.0f} KB" if size < 1_000_000 else f"{size/1_000_000:.1f} MB"
            print(f"  ✅  {fname:<42} [{size_str}]")
            print(f"      └─ {desc}")
        else:
            print(f"  ⚪  {fname:<42} [not generated]")

    print(f"\n{'═'*WIDTH}")
    print("  🚀  NEXT STEPS:")
    print("  1. Open results/bangalore_interactive_map.html in your browser")
    print("  2. Toggle the layer controls (top-right) to compare maps")
    print("  3. Use the ❓ Help button on the map for a guided tour")
    print(f"{'═'*WIDTH}\n")


if __name__ == "__main__":
    banner()
    Path("results").mkdir(exist_ok=True)

    for i, step in enumerate(STEPS, start=1):
        success = run_step(i, step)
        if not success:
            sys.exit(1)

    print(f"\n{'═'*WIDTH}")
    print("  🎉  PIPELINE COMPLETE — All steps finished!")
    print(f"{'═'*WIDTH}")
    results_summary()
