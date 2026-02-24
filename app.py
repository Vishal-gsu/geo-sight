"""
app.py — GeoSight Streamlit Dashboard

Single-command launch:
    streamlit run app.py

User only needs to set EE_PROJECT_ID in their .env file first.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import streamlit as st
import folium
from pathlib import Path
from dotenv import load_dotenv
from streamlit_folium import st_folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoSight — Satellite Analysis Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load .env & GEE ──────────────────────────────────────────────────────────
load_dotenv()
PROJECT_ID = os.environ.get("EE_PROJECT_ID", "")

@st.cache_resource(show_spinner=False)
def init_gee(project_id):
    import ee
    try:
        ee.Initialize(project=project_id)
        return ee, True, None
    except Exception as e:
        return None, False, str(e)

# ── City registry ─────────────────────────────────────────────────────────────
CITIES = {
    "Bengaluru 🌿": {
        "center": [12.9716, 77.5946],
        "roi":    [77.45, 12.85, 77.75, 13.10],
        "desc":   "India's IT capital — rapid urban expansion into green corridors since 2015"
    },
    "Mumbai 🏙️": {
        "center": [19.0760, 72.8777],
        "roi":    [72.77, 18.90, 73.02, 19.25],
        "desc":   "Coastal megacity — mangrove loss and reclamation visible from space"
    },
    "Delhi 🌫️": {
        "center": [28.6139, 77.2090],
        "roi":    [76.95, 28.40, 77.50, 28.85],
        "desc":   "NCT — agricultural land conversion and rapid satellite city growth"
    },
    "Chennai 🌊": {
        "center": [13.0827, 80.2707],
        "roi":    [80.14, 12.90, 80.40, 13.25],
        "desc":   "Coastal city — wetland encroachment and urban heat island analysis"
    },
    "Hyderabad 💻": {
        "center": [17.3850, 78.4867],
        "roi":    [78.30, 17.25, 78.70, 17.60],
        "desc":   "Pharma & tech hub — HITEC city expansion into Musi River flood plains"
    },
    "Pune 🏔️": {
        "center": [18.5204, 73.8567],
        "roi":    [73.70, 18.40, 74.00, 18.65],
        "desc":   "Growing metro — Western Ghats buffer zone pressure and IT corridor growth"
    },
}

INDICES = {
    "NDVI — Vegetation Health": {
        "bands": ["B8", "B4"],
        "palette": ["#d73027", "#ffffbf", "#1a9850"],
        "range":   [-0.2, 0.8],
        "desc":    "Higher = denser vegetation. Formula: (NIR–Red)/(NIR+Red)"
    },
    "NDWI — Water Bodies": {
        "bands": ["B3", "B8"],
        "palette": ["#ffffb2", "#74c476", "#08519c"],
        "range":   [-0.3, 0.5],
        "desc":    "Higher = open water. Formula: (Green–NIR)/(Green+NIR)"
    },
    "NDBI — Built-up / Urban": {
        "bands": ["B11", "B8"],
        "palette": ["#ffffff", "#fd8d3c", "#bd0026"],
        "range":   [-0.3, 0.5],
        "desc":    "Higher = more built-up area. Formula: (SWIR–NIR)/(SWIR+NIR)"
    },
}

RESULTS_DIR = Path("results")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .metric-card {
    background: linear-gradient(135deg, rgba(125,211,252,0.08), rgba(99,102,241,0.08));
    border: 1px solid rgba(125,211,252,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-card .val { font-size: 2em; font-weight: 700; }
  .metric-card .lbl { font-size: 0.75em; color: #94a3b8; margin-top: 4px; }
  .stButton > button {
    background: linear-gradient(135deg, #7dd3fc, #6366f1);
    color: white; font-weight: 600; border: none;
    border-radius: 8px; padding: 10px 28px;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }
  div[data-testid="stStatusWidget"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰️ GeoSight")
    st.caption("Satellite Land Cover Analysis")
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Home",
        "📊 Change Detection",
        "🗺️ Interactive Map",
        "🤖 SAM Segmentation",
        "📁 Results Gallery",
    ], label_visibility="collapsed")

    st.divider()

    # GEE status
    if PROJECT_ID:
        ee, gee_ok, gee_err = init_gee(PROJECT_ID)
        if gee_ok:
            st.success(f"✅ GEE Connected\n`{PROJECT_ID}`")
        else:
            st.error(f"❌ GEE Error\n{gee_err[:80]}")
            st.caption("Run `earthengine authenticate` in terminal")
    else:
        st.warning("⚠️ Add `EE_PROJECT_ID` to your `.env` file")
        gee_ok = False
        ee = None

    st.divider()
    st.caption("CMR Institute of Technology\nSentinel-2 L2A · Google Earth Engine")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("# 🛰️ GeoSight")
    st.markdown("### Satellite Land Cover Change Detection Dashboard")
    st.markdown("*Powered by Google Earth Engine · Sentinel-2 · Meta SAM · Python*")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="val">6</div><div class="lbl">Cities Supported</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="val">7yr</div><div class="lbl">Data Range (2018–2024)</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="val">3</div><div class="lbl">Spectral Indices</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="val">10m</div><div class="lbl">Ground Resolution</div></div>', unsafe_allow_html=True)

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 📖 How to Use")
        st.markdown("""
1. **Set your `.env`** — add `EE_PROJECT_ID=your-project-id`
2. **Check sidebar** — GEE Connected status should be ✅
3. **Change Detection** — pick a city & years → click Run
4. **Interactive Map** — toggle satellite layers live
5. **SAM Segmentation** — zero-shot land cover map
6. **Results Gallery** — all generated images in one place
        """)

    with col_b:
        st.markdown("### 🔬 What This Analyses")
        for idx_name, idx in INDICES.items():
            st.markdown(f"**{idx_name}**")
            st.caption(idx["desc"])

    st.divider()
    st.markdown("### 🌆 Supported Cities")
    cols = st.columns(3)
    for i, (city, info) in enumerate(CITIES.items()):
        with cols[i % 3]:
            st.info(f"**{city}**\n\n{info['desc']}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CHANGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Change Detection":
    st.markdown("## 📊 Change Detection")
    st.markdown("Compare land cover changes between two years using real Sentinel-2 data.")

    if not gee_ok:
        st.error("🔴 Earth Engine not connected. Check sidebar.")
        st.stop()

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        city_name = st.selectbox("🌆 Select City", list(CITIES.keys()))
    with col2:
        year_t1 = st.selectbox("📅 Year T1 (Before)", range(2018, 2025), index=1)
    with col3:
        year_t2 = st.selectbox("📅 Year T2 (After)", range(2018, 2025), index=6)

    idx_name = st.selectbox("📡 Spectral Index", list(INDICES.keys()))
    city = CITIES[city_name]
    idx  = INDICES[idx_name]

    st.info(f"**{city_name}:** {city['desc']}")

    if st.button("🚀 Run Analysis", use_container_width=True):
        if year_t1 >= year_t2:
            st.error("T1 must be before T2!")
            st.stop()

        roi_coords = city["roi"]
        roi = ee.Geometry.Rectangle(roi_coords)
        short_name = idx_name.split("—")[0].strip()

        with st.spinner(f"Fetching Sentinel-2 composites for {city_name} ({year_t1} & {year_t2})…"):
            def get_composite(year):
                return (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(roi)
                    .filterDate(f"{year}-01-01", f"{year}-12-31")
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15))
                    .select(["B2","B3","B4","B8","B11"])
                    .median()
                )
            img_t1 = get_composite(year_t1)
            img_t2 = get_composite(year_t2)

            b1, b2 = idx["bands"]
            index_t1 = img_t1.normalizedDifference([b1, b2]).rename(short_name)
            index_t2 = img_t2.normalizedDifference([b1, b2]).rename(short_name)
            delta    = index_t2.subtract(index_t1).rename(f"d{short_name}")

        with st.spinner("Computing regional statistics…"):
            SCALE = 100
            mean_t1 = list(index_t1.reduceRegion(ee.Reducer.mean(), roi, SCALE).getInfo().values())[0]
            mean_t2 = list(index_t2.reduceRegion(ee.Reducer.mean(), roi, SCALE).getInfo().values())[0]
            d_val   = mean_t2 - mean_t1

        # Show stats
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{short_name} {year_t1}", f"{mean_t1:.4f}")
        c2.metric(f"{short_name} {year_t2}", f"{mean_t2:.4f}")
        c3.metric(f"Δ Change", f"{d_val:+.4f}",
                  delta=f"{'↓ Loss' if d_val < 0 else '↑ Gain'}",
                  delta_color="inverse" if short_name == "NDVI" else "normal")

        # Interpretation
        if short_name == "NDVI":
            if d_val < -0.02:
                st.warning(f"⚠️ **Vegetation declining** in {city_name.split()[0]} — NDVI dropped {d_val:.4f} from {year_t1} to {year_t2}. Likely urban expansion or deforestation.")
            elif d_val > 0.02:
                st.success(f"🌿 **Vegetation improving** — NDVI rose {d_val:+.4f}. Possible afforestation or seasonal recovery.")
            else:
                st.info(f"📊 **Vegetation stable** — NDVI change of {d_val:+.4f} is within natural seasonal variation.")
        elif short_name == "NDBI":
            if d_val > 0.01:
                st.warning(f"🏙️ **Urban expansion detected** — NDBI increased {d_val:+.4f} suggesting new construction.")
        elif short_name == "NDWI":
            if d_val < -0.01:
                st.warning(f"💧 **Water bodies shrinking** — NDWI dropped {d_val:.4f}. Possible lake encroachment.")

        # Live map
        st.divider()
        st.markdown(f"#### 🗺️ {short_name} Change Map — {year_t1} → {year_t2}")

        with st.spinner("Generating map tiles from GEE…"):
            vis_base = {"min": idx["range"][0], "max": idx["range"][1], "palette": idx["palette"]}
            vis_delta = {"min": -0.3, "max": 0.3, "palette": ["#d73027", "#ffffff", "#1a9850"]}

            m = folium.Map(location=city["center"], zoom_start=11, tiles="CartoDB dark_matter")
            for img, vis, name in [
                (index_t1, vis_base, f"{short_name} {year_t1}"),
                (index_t2, vis_base, f"{short_name} {year_t2}"),
                (delta,    vis_delta, f"Δ Change (Red=Loss)"),
            ]:
                url = img.getMapId(vis)["tile_fetcher"].url_format
                folium.TileLayer(tiles=url, attr="GEE", name=name, overlay=True, control=True).add_to(m)
            folium.LayerControl(collapsed=False, position="topright").add_to(m)

        st_folium(m, height=480, use_container_width=True)

        # Save stats
        RESULTS_DIR.mkdir(exist_ok=True)
        stats_out = {
            "city": city_name, "index": short_name,
            "year_t1": year_t1, "year_t2": year_t2,
            "mean_t1": round(mean_t1, 4), "mean_t2": round(mean_t2, 4),
            "delta": round(d_val, 4)
        }
        with open(RESULTS_DIR / "last_analysis.json", "w") as f:
            json.dump(stats_out, f, indent=2)
        st.success("✅ Results saved to `results/last_analysis.json`")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INTERACTIVE MAP  (load-on-demand, cached across tab switches)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗺️ Interactive Map":
    import streamlit.components.v1 as components

    # ── persist map state so tab-switching doesn't reload ────────────────────
    if "_map_html" not in st.session_state:
        st.session_state._map_html      = None   # rendered HTML string
        st.session_state._map_label     = None   # e.g. "Mumbai 2019→2024"
        st.session_state._map_loading   = False

    st.markdown("## 🗺️ Interactive Satellite Map")

    if not gee_ok:
        st.error("🔴 Earth Engine not connected. Check sidebar.")
        st.stop()

    # ── Controls row ─────────────────────────────────────────────────────────
    cc1, cc2, cc3, cc4 = st.columns([3, 1, 1, 1])
    with cc1:
        map_city_name = st.selectbox("🌆 City", list(CITIES.keys()), key="map_city")
    with cc2:
        map_yr1 = st.selectbox("T1", range(2018, 2025), index=1, key="map_yr1")
    with cc3:
        map_yr2 = st.selectbox("T2", range(2018, 2025), index=6, key="map_yr2")
    with cc4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        load_clicked = st.button("🗺️ Load Map", use_container_width=True,
                                 type="primary",
                                 disabled=st.session_state._map_loading)

    map_city = CITIES[map_city_name]
    st.caption(f"📍 {map_city['desc']}")
    st.divider()

    tab_map, tab_guide = st.tabs(["🌐 Map", "ℹ️ Layer Guide"])

    with tab_map:
        if load_clicked:
            st.session_state._map_loading = True
            label = f"{map_city_name.split()[0]} {map_yr1}→{map_yr2}"
            with st.spinner(f"Fetching GEE tiles for {label}…  (~10 sec)"):
                roi = ee.Geometry.Rectangle(map_city["roi"])

                def _comp(yr):
                    return (
                        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(roi)
                        .filterDate(f"{yr}-01-01", f"{yr}-12-31")
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15))
                        .select(["B2","B3","B4","B8","B11"]).median()
                    )

                c1, c2 = _comp(map_yr1), _comp(map_yr2)
                ndvi1  = c1.normalizedDifference(["B8","B4"]).rename("NDVI")
                ndvi2  = c2.normalizedDifference(["B8","B4"]).rename("NDVI")
                ndbi2  = c2.normalizedDifference(["B11","B8"]).rename("NDBI")
                ndwi2  = c2.normalizedDifference(["B3","B8"]).rename("NDWI")
                d_ndvi = ndvi2.subtract(ndvi1).rename("dNDVI")
                d_ndbi = ndbi2.subtract(c1.normalizedDifference(["B11","B8"])).rename("dNDBI")

                m = folium.Map(location=map_city["center"], zoom_start=11,
                               tiles="CartoDB dark_matter")

                LAYERS = [
                    (ndvi1,  {"min":-0.2,"max":0.8, "palette":["#d73027","#ffffbf","#1a9850"]}, f"🌿 NDVI {map_yr1}"),
                    (ndvi2,  {"min":-0.2,"max":0.8, "palette":["#d73027","#ffffbf","#1a9850"]}, f"🌿 NDVI {map_yr2}"),
                    (d_ndvi, {"min":-0.4,"max":0.4, "palette":["#d73027","#ffffff","#1a9850"]}, "🔴 ΔNDVI Change"),
                    (ndbi2,  {"min":-0.3,"max":0.5, "palette":["#ffffff","#fd8d3c","#bd0026"]}, f"🏙️ NDBI {map_yr2}"),
                    (d_ndbi, {"min":-0.3,"max":0.5, "palette":["#ffffff","#fd8d3c","#bd0026"]}, "🏗️ ΔNDBI Growth"),
                    (ndwi2,  {"min":-0.3,"max":0.5, "palette":["#ffffb2","#74c476","#08519c"]}, f"💧 NDWI {map_yr2}"),
                ]
                for img, vis, name in LAYERS:
                    url = img.getMapId(vis)["tile_fetcher"].url_format
                    # show=False → all layers unchecked by default, user picks what to see
                    folium.TileLayer(
                        tiles=url, attr="GEE", name=name,
                        overlay=True, control=True, opacity=0.85, show=False
                    ).add_to(m)

                folium.LayerControl(collapsed=False, position="topright").add_to(m)

                import io
                html_io = io.BytesIO()
                m.save(html_io, close_file=False)
                st.session_state._map_html    = html_io.getvalue().decode("utf-8")
                st.session_state._map_label   = label
                st.session_state._map_loading = False

        if st.session_state._map_html:
            st.markdown(
                f"<div style='display:inline-block;background:rgba(125,211,252,0.1);"
                f"border:1px solid rgba(125,211,252,0.3);border-radius:8px;"
                f"padding:4px 12px;font-size:12px;margin-bottom:8px'>"
                f"📍 <b>{st.session_state._map_label}</b> loaded "
                f"· Tick layers (top-right) to reveal them</div>",
                unsafe_allow_html=True
            )
            components.html(st.session_state._map_html, height=620, scrolling=False)
            if st.button("🔄 Reload with current settings"):
                st.session_state._map_html  = None
                st.session_state._map_label = None
                st.rerun()
        else:
            # Pretty placeholder before first load
            st.markdown("""
<div style="border:1px dashed rgba(125,211,252,0.3);border-radius:16px;
            padding:60px 40px;text-align:center;margin-top:16px">
  <div style="font-size:48px;margin-bottom:12px">🛰️</div>
  <div style="font-size:18px;font-weight:600;color:#7dd3fc;margin-bottom:8px">
    Select City + Years → Press Load Map
  </div>
  <div style="font-size:13px;color:#64748b">
    All 6 satellite layers will load. Tick layers individually to compare.<br>
    Map stays cached — switch tabs freely without reloading.
  </div>
</div>""", unsafe_allow_html=True)

    with tab_guide:
        st.markdown("### What each layer shows")
        layer_guide = [
            ("🌿 NDVI — Vegetation", ["#d73027","#ffffbf","#1a9850"],
             ["Bare/Urban","Sparse","Dense Vegetation"],
             "NIR bands detect plant chlorophyll invisible to human eye. High NDVI = healthy forest/parks."),
            ("🔴 ΔNDVI Change", ["#d73027","#ffffff","#1a9850"],
             ["Vegetation lost","No change","Vegetation gained"],
             "Subtract T1 from T2. Red = deforestation, construction. Green = new growth."),
            ("🏙️ NDBI — Urban Built-up", ["#ffffff","#fd8d3c","#bd0026"],
             ["No urban","Moderate","Dense urban"],
             "Concrete & asphalt reflect SWIR strongly. High NDBI = buildings, roads."),
            ("🏗️ ΔNDBI Growth", ["#ffffff","#fd8d3c","#bd0026"],
             ["No change","Some growth","High growth"],
             "Where new concrete appeared between T1 and T2 — spot new suburbs, highways."),
            ("💧 NDWI — Water", ["#ffffb2","#74c476","#08519c"],
             ["No water","Moist","Open water"],
             "Green band absorbed by water. Blue = lakes, rivers. Track shrinkage over time."),
        ]
        for name, colors, labels, desc in layer_guide:
            with st.expander(f"**{name}**"):
                st.caption(desc)
                c1c, c2c, c3c = st.columns(3)
                for col_obj, clr, lbl in zip([c1c, c2c, c3c], colors, labels):
                    with col_obj:
                        st.markdown(
                            f'<div style="background:{clr};height:28px;border-radius:6px;margin-bottom:4px"></div>'
                            f'<div style="font-size:11px;text-align:center">{lbl}</div>',
                            unsafe_allow_html=True
                        )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SAM SEGMENTATION  (full state persisted in session_state)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 SAM Segmentation":
    import threading

    # ── ALL page state lives in session_state so tab-switching is lossless ───
    SAM_KEYS = {
        "_sam_proc_thread":  None,
        "_sam_proc_status":  "idle",    # idle | downloading | running | done | error
        "_sam_proc_city":    list(CITIES.keys())[0],
        "_sam_proc_log":     "",
    }
    for k, v in SAM_KEYS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    st.markdown("## 🤖 SAM Segmentation")
    st.markdown("Zero-shot land cover detection using Meta's Segment Anything Model `vit_b`.")

    # City selector — index driven by persisted _sam_proc_city
    city_list = list(CITIES.keys())
    saved_city = st.session_state._sam_proc_city
    saved_idx  = city_list.index(saved_city) if saved_city in city_list else 0
    city_name  = st.selectbox("🌆 City", city_list, index=saved_idx, key="sam_city_sel")

    # immediately persist selection so tab switch keeps it
    st.session_state._sam_proc_city = city_name

    city_key  = city_name.split()[0]
    city      = CITIES[city_name]
    slug      = city_key.lower()
    tile_path = Path(f"data/{slug}_patch/{slug}_rgb.png")

    st.info(f"**{city_name}:** {city['desc']}")

    # ── Status banner (always visible even after tab switch) ─────────────────
    status = st.session_state._sam_proc_status
    if status == "downloading":
        st.info("📡 **Tile download running in background.** Switch tabs freely — click Check Status when back.")
        if st.button("🔄 Check Download Status"):
            t = st.session_state._sam_proc_thread
            if t and not t.is_alive():
                st.session_state._sam_proc_status = "idle"
            st.rerun()
    elif status == "running":
        st.warning("⏳ **SAM running in background (~2 min).** Switch tabs freely — click Check Status when back.")
        if st.button("🔄 Check SAM Status"):
            t = st.session_state._sam_proc_thread
            if t and not t.is_alive():
                st.session_state._sam_proc_status = "done"
            st.rerun()
    elif status == "done":
        st.success("✅ SAM segmentation complete! See results below ↓")
    elif status == "error":
        st.error("❌ Process failed.")
        with st.expander("Error log"):
            st.code(st.session_state._sam_proc_log[-1200:])

    st.divider()
    col1, col2 = st.columns(2)

    # ── Step 1: Download ─────────────────────────────────────────────────────
    with col1:
        st.markdown("#### 📥 Step 1 — Download Tile")
        st.caption("Real 2048×2048 Sentinel-2 RGB patch from GEE · ~30 sec")
        busy = status in ("downloading", "running")

        if st.button("📥 Download Tile", use_container_width=True, disabled=busy, key="dl_btn"):
            st.session_state._sam_proc_status = "downloading"

            def _dl(ck=city_key):
                r = subprocess.run(
                    [sys.executable, "download_tile.py", "--city", ck],
                    capture_output=True, text=True, cwd=Path(__file__).parent
                )
                st.session_state._sam_proc_log    = r.stdout + r.stderr
                st.session_state._sam_proc_status = "idle" if r.returncode == 0 else "error"

            t = threading.Thread(target=_dl, daemon=True)
            t.start()
            st.session_state._sam_proc_thread = t
            st.rerun()

        if tile_path.exists():
            st.success(f"✅ `{tile_path.name}` ready")
            st.image(str(tile_path),
                     caption=f"{city_name} · 2048×2048 · Sentinel-2",
                     width='stretch')
        else:
            st.markdown(
                '<div style="border:1px dashed rgba(125,211,252,0.25);border-radius:12px;'
                'padding:40px;text-align:center;color:#475569">'
                '📡 No tile downloaded yet.<br>Click Download Tile above.</div>',
                unsafe_allow_html=True
            )

    # ── Step 2: SAM ──────────────────────────────────────────────────────────
    with col2:
        st.markdown("#### 🤖 Step 2 — Run SAM")
        st.caption("Auto-detects all land cover segments · ~2 min on CPU")
        sam_ready = tile_path.exists() and status not in ("running", "downloading")

        if st.button("🤖 Run SAM", use_container_width=True,
                     disabled=not sam_ready, key="sam_btn"):
            st.session_state._sam_proc_status = "running"

            def _sam(tp=str(tile_path), ck=city_key):
                r = subprocess.run(
                    [sys.executable, "sam2_segmentation.py", "--input", tp, "--city", ck],
                    capture_output=True, text=True, cwd=Path(__file__).parent
                )
                st.session_state._sam_proc_log    = r.stdout + r.stderr
                st.session_state._sam_proc_status = "done" if r.returncode == 0 else "error"

            t = threading.Thread(target=_sam, daemon=True)
            t.start()
            st.session_state._sam_proc_thread = t
            st.rerun()

        # Show results preview in col2
        summary = Path("results/bangalore_sam2_summary.png")
        if summary.exists():
            st.image(str(summary), caption="3-Panel: RGB → SAM → Classes", width='stretch')
        else:
            st.markdown(
                '<div style="border:1px dashed rgba(125,211,252,0.25);border-radius:12px;'
                'padding:40px;text-align:center;color:#475569">'
                '🤖 No segmentation yet.<br>Download tile → Run SAM.</div>',
                unsafe_allow_html=True
            )

    # ── Results gallery ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 All Segmentation Outputs")
    sam_result_files = {
        "results/bangalore_sam2_summary.png":            "3-Panel: RGB → SAM Segments → Land Cover",
        "results/bangalore_class_overlay.png":           "Land cover classes on satellite image",
        "results/bangalore_classification_map.png":      "Pure classification map",
        "results/bangalore_class_overlay_fallback.png":  "Spectral fallback classification",
    }
    any_shown = False
    for fpath, cap in sam_result_files.items():
        p = Path(fpath)
        if p.exists():
            any_shown = True
            with st.expander(f"📊 {cap}", expanded=False):
                st.image(str(p), width='stretch')

    for sp in [Path("results/bangalore_class_stats.json"),
               Path("results/bangalore_class_stats_fallback.json")]:
        if sp.exists():
            with open(sp) as f:
                sd = json.load(f)
            st.markdown("#### Area Statistics")
            try:
                sc = st.columns(len(sd))
                for i, (cls, s) in enumerate(sd.items()):
                    sc[i].metric(cls, f"{s['area_sq_km']:.2f} km²", f"{s['percentage']:.1f}%")
            except Exception:
                st.json(sd)
            break

    if not any_shown:
        st.info("No results yet — complete Step 1 then Step 2 above.")



# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RESULTS GALLERY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📁 Results Gallery":
    st.markdown("## 📁 Results Gallery")
    st.markdown("All outputs generated by GeoSight pipeline are displayed here.")

    RESULTS_DIR.mkdir(exist_ok=True)
    all_images = sorted(RESULTS_DIR.glob("*.png"))

    descriptions = {
        "demo_output.png":                 "Synthetic demo — NDVI T1/T2 + change mask + deforestation detection on mock data",
        "bangalore_change_detection.png":  "6-panel scientific figure — NDVI Before/After/Δ | NDBI Before/After | Change Map",
        "bangalore_sam2_summary.png":      "SAM pipeline — raw satellite patch → detected segments → classified land cover",
        "bangalore_class_overlay.png":     "SAM land cover classes painted transparently over real satellite image",
        "bangalore_classification_map.png":"Pure classification map — each colour = a land cover class (no base layer)",
        "bangalore_rgb_sam2_overlay.png":  "All SAM-detected object segments shown in random colours over RGB image",
        "bangalore_class_overlay_fallback.png": "Fallback: pure spectral classification (no SAM) — used when SAM is unavailable",
        "bangalore_classification_fallback.png":"Fallback classification map from spectral thresholds only",
    }

    if not all_images:
        st.info("No images generated yet. Run Change Detection or SAM Segmentation first.")
        st.code("python bangalore_change_detection.py\npython sam2_segmentation.py", language="bash")
    else:
        for img_path in all_images:
            name = img_path.name
            desc = descriptions.get(name, "Generated output image")
            with st.expander(f"📊 **{name}** — {desc}", expanded="sam2_summary" in name or "change_detection" in name):
                img = Image.open(img_path)
                st.image(img, width='stretch')
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"📁 `results/{name}` · {img.size[0]}×{img.size[1]}px · {img_path.stat().st_size/1024:.0f} KB")
                with col2:
                    with open(img_path, "rb") as f:
                        st.download_button("⬇️ Download", f, file_name=name, mime="image/png", use_container_width=True)

    # JSON stats
    json_files = list(RESULTS_DIR.glob("*.json"))
    if json_files:
        st.divider()
        st.markdown("### 📋 Statistics Files")
        for jf in json_files:
            with st.expander(f"📋 `{jf.name}`"):
                with open(jf) as f:
                    data = json.load(f)
                st.json(data)
