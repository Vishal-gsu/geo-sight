"""
report.py

Production module for visual output generation, area quantification,
interactive map creation (Folium), and PDF/PNG report export.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium

logger = logging.getLogger(__name__)

# Class definitions (matches segmentation.py)
CLASS_COLORS_HEX = {
    "Vegetation": "#228B22",
    "Water":      "#4169E1",
    "Urban":      "#FF8C00",
    "Bare Land":  "#D2B48C",
    "Unknown":    "#808080",
}

CLASS_IDS = {0: "Unknown", 1: "Vegetation", 2: "Water", 3: "Urban", 4: "Bare Land"}


# ──────────────────────────────────────────────────────────
# 1. Static Matplotlib Figures
# ──────────────────────────────────────────────────────────

def plot_rgb(image_rgb: np.ndarray, title: str = "RGB Preview", out_path: Optional[str] = None) -> None:
    """Saves a true-colour RGB preview of the satellite patch."""
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0f1117")
    ax.imshow(image_rgb, interpolation="bilinear")
    ax.set_title(title, color="white", fontsize=12, pad=6)
    ax.axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"✅ RGB preview saved: {out_path}")
    plt.close()


def plot_spectral_index(index_array: np.ndarray, title: str, cmap: str = "RdYlGn",
                        vmin: float = -1.0, vmax: float = 1.0, out_path: Optional[str] = None) -> None:
    """Saves a single spectral index heatmap (NDVI, NDWI, or NDBI)."""
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0f1117")
    im = ax.imshow(index_array, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_title(title, color="white", fontsize=12, pad=6)
    ax.axis("off")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="gray")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"✅ Spectral index plot saved: {out_path}")
    plt.close()


def plot_classification_map(class_map: np.ndarray, out_path: Optional[str] = None) -> None:
    """Renders a land-cover classification map with a categorical legend."""
    import matplotlib.colors as mcolors

    n_classes = len(CLASS_IDS)
    palette   = [CLASS_COLORS_HEX[CLASS_IDS[i]] for i in range(n_classes)]
    cmap      = mcolors.ListedColormap(palette)
    bounds    = list(range(n_classes + 1))
    norm      = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0f1117")
    ax.imshow(class_map, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title("Land Cover Classification\n(SAM 2 + Spectral Index Fusion)", color="white", fontsize=12, pad=8)
    ax.axis("off")

    patches = [mpatches.Patch(color=CLASS_COLORS_HEX[name], label=name)
               for name in CLASS_IDS.values()]
    ax.legend(handles=patches, loc="lower right", framealpha=0.7,
              facecolor="#1a1d27", labelcolor="white", fontsize=9)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"✅ Classification map saved: {out_path}")
    plt.close()


def plot_change_comparison(
    ndvi_t1: np.ndarray, ndvi_t2: np.ndarray,
    delta_ndvi: np.ndarray, defor_mask: np.ndarray,
    out_path: Optional[str] = None
) -> None:
    """2×2 panel figure showing NDVI before/after, delta, and binary deforestation mask."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.axis("off")

    kw = dict(vmin=-0.2, vmax=0.8, cmap="RdYlGn", interpolation="bilinear")
    axes[0].imshow(ndvi_t1,    **kw); axes[0].set_title("NDVI — T1", color="white")
    axes[1].imshow(ndvi_t2,    **kw); axes[1].set_title("NDVI — T2", color="white")
    axes[2].imshow(delta_ndvi, vmin=-0.4, vmax=0.4, cmap="RdYlGn_r",
                   interpolation="bilinear"); axes[2].set_title("ΔNDVI", color="white")
    axes[3].imshow(defor_mask, cmap="Reds", vmin=0, vmax=1,
                   interpolation="none"); axes[3].set_title("Deforestation Mask", color="white")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"✅ Change comparison saved: {out_path}")
    plt.close()


# ──────────────────────────────────────────────────────────
# 2. Area Statistics
# ──────────────────────────────────────────────────────────

def generate_area_stats(mask_array: np.ndarray, pixel_resolution_m: float = 10.0) -> dict:
    """
    Converts a binary pixel mask into physical area statistics.

    Args:
        mask_array          : (H, W) uint8 or bool — 1 = positive class
        pixel_resolution_m  : ground sampling distance (10m for Sentinel-2 at 10m res)

    Returns:
        dict with 'pixels', 'area_sq_m', 'area_sq_km'
    """
    total_pixels = int(mask_array.sum())
    area_sq_m    = total_pixels * (pixel_resolution_m ** 2)
    area_sq_km   = area_sq_m / 1_000_000.0
    logger.info(f"Area stats: {total_pixels} px = {area_sq_km:.2f} km²")
    return {"pixels": total_pixels, "area_sq_m": area_sq_m, "area_sq_km": round(area_sq_km, 4)}


def generate_class_area_report(class_map: np.ndarray, pixel_resolution_m: float = 10.0) -> dict:
    """
    Computes area per land-cover class from a classification map.

    Returns:
        dict keyed by class name, values contain pixel count and area in km²
    """
    report = {}
    total  = class_map.size
    for class_id, class_name in CLASS_IDS.items():
        mask  = (class_map == class_id)
        stats = generate_area_stats(mask, pixel_resolution_m)
        stats["percentage"] = round(mask.sum() / total * 100, 2)
        report[class_name] = stats
    return report


def save_stats_json(stats: dict, output_path: str) -> None:
    """Saves a statistics dictionary to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✅ Stats saved: {output_path}")


# ──────────────────────────────────────────────────────────
# 3. Interactive Folium Map
# ──────────────────────────────────────────────────────────

def create_html_map(
    center_latlon: list,
    output_path: str,
    gee_tile_layers: Optional[list] = None,
    zoom_start: int = 11
) -> None:
    """
    Creates an interactive Folium HTML map with optional GEE tile overlays.

    Args:
        center_latlon   : [lat, lon] of map center (e.g. Bengaluru: [12.97, 77.59])
        output_path     : path to save the .html file
        gee_tile_layers : list of dicts with keys 'url', 'name', 'opacity'
        zoom_start      : initial zoom level
    """
    m = folium.Map(location=center_latlon, zoom_start=zoom_start, tiles="CartoDB dark_matter")

    if gee_tile_layers:
        for layer in gee_tile_layers:
            folium.TileLayer(
                tiles=layer["url"],
                attr="Google Earth Engine",
                name=layer["name"],
                overlay=True,
                control=True,
                opacity=layer.get("opacity", 0.8)
            ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Info panel
    info_html = f"""
    <div style="position:fixed;top:10px;left:60px;z-index:9999;
                background:#111522;color:white;padding:10px 16px;
                border-radius:8px;font-size:13px;border:1px solid #334;max-width:260px">
      <b>🛰️ GeoSight — Bengaluru</b><br>
      <span style="color:#aaa;font-size:11px">Sentinel-2 · Land Cover Analysis</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(output_path)
    logger.info(f"✅ Interactive map saved: {output_path}")
