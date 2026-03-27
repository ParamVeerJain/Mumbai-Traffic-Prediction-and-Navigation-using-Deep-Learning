"""
config.py  —  All constants for the Mumbai traffic dataset pipeline.
"""

import numpy as np

# ─── OSM Grid ────────────────────────────────────────────────────────────────
LATS         = np.arange(19.00, 19.45, 0.05)   # 9 centre latitudes
LONS         = np.arange(72.80, 73.10, 0.05)   # 6 centre longitudes
RADIUS_M     = 6_000                             # metres per centre point
NETWORK_TYPE = "drive"

# ─── Time Series ─────────────────────────────────────────────────────────────
WEEK_START = "2024-07-01 00:00:00"   # Monday  —  peak Mumbai monsoon season
N_HOURS    = 168                      # 7 days × 24 hrs

# ─── Generation ──────────────────────────────────────────────────────────────
BATCH_SIZE = 5_000          # edges per batch (keeps RAM <200 MB per batch)
FIELD_RES  = 0.02           # degrees per congestion-field cell (≈ 2.2 km)
RANDOM_SEED = 42

# ─── Road type → free-flow speed (km/h) ─────────────────────────────────────
FREE_FLOW_SPEED = {
    "motorway": 100, "motorway_link": 80,
    "trunk":     80, "trunk_link":    70,
    "primary":   60, "primary_link":  50,
    "secondary": 50, "secondary_link":45,
    "tertiary":  40, "tertiary_link": 35,
    "residential":30, "living_street":15,
    "service":   20, "unclassified":  30,
}

# ─── Road type → default lanes ───────────────────────────────────────────────
LANE_DEFAULTS = {
    "motorway":6, "motorway_link":2, "trunk":4, "trunk_link":2,
    "primary":2, "primary_link":1, "secondary":2, "secondary_link":1,
    "tertiary":1, "tertiary_link":1, "residential":1,
    "living_street":1, "service":1, "unclassified":1,
}

# ─── Road type → ordinal encoding (higher = more capacity) ──────────────────
ROAD_TYPE_ENC = {
    "motorway":8, "motorway_link":7, "trunk":7, "trunk_link":6,
    "primary":6, "primary_link":5, "secondary":5, "secondary_link":4,
    "tertiary":4, "tertiary_link":3, "residential":2,
    "living_street":1, "service":1, "unclassified":1,
}

# ─── Road type → congestion susceptibility (1.0 = neutral) ──────────────────
# How much zone-level congestion propagates to this road type.
# High-capacity roads buffer congestion; narrow secondary roads amplify it.
SUSCEPTIBILITY = {
    "motorway":     0.78, "motorway_link":0.85,
    "trunk":        0.88, "trunk_link":   0.90,
    "primary":      1.05, "primary_link": 1.00,
    "secondary":    1.10, "secondary_link":1.05,
    "tertiary":     1.00, "tertiary_link":0.95,
    "residential":  0.70, "living_street":0.55,
    "service":      0.50, "unclassified": 0.75,
}

# ─── TomTom incident iconCategory subset used ───────────────────────────────
# 0  = no incident  (internal convention; TomTom itself never returns 0)
# 1  = Accident
# 4  = Rain
# 6  = Jam
# 8  = Road Closed
# 9  = Road Works
# 11 = Flooding
INC_TYPES = [1, 4, 6, 8, 9, 11]

# ─── Directories ─────────────────────────────────────────────────────────────
DATA_DIR       = "data"
GRAPH_DIR      = f"{DATA_DIR}/graph"
STATIC_DIR     = f"{DATA_DIR}/static"
TIMESERIES_DIR = f"{DATA_DIR}/timeseries"
