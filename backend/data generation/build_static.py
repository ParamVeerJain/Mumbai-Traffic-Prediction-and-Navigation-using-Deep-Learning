"""
build_static.py
───────────────
Loads the saved OSM GraphML, extracts one static-feature row per road edge,
and writes  data/static/edges_static.parquet.

These features never change over time; they are joined to the time-series
batches via  edge_id  at training time.

Run:
    python build_static.py
"""

import os
import numpy as np
import pandas as pd
import osmnx as ox
from config import (
    FREE_FLOW_SPEED, LANE_DEFAULTS, ROAD_TYPE_ENC,
    SUSCEPTIBILITY, FIELD_RES, GRAPH_DIR, STATIC_DIR,
)


# ─── small helpers ────────────────────────────────────────────────────────────

def _first(v, default=None):
    """OSM tag values can be lists; take the first element."""
    if isinstance(v, list):
        return v[0] if v else default
    return v


def _parse_maxspeed(v) -> float:
    v = _first(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    try:
        return float(str(v).lower()
                     .replace("km/h", "").replace("mph", "")
                     .replace("kph", "").strip())
    except ValueError:
        return np.nan


def _parse_lanes(v) -> float:
    v = _first(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    try:
        return float(str(v).strip())
    except ValueError:
        return np.nan


def _normalise_road_type(v) -> str:
    v = _first(v, "unclassified")
    if not isinstance(v, str):
        return "unclassified"
    v = v.strip().lower()
    return v if v in FREE_FLOW_SPEED else "unclassified"


def _parse_oneway(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "yes", "1")
    return False


# ─── main ─────────────────────────────────────────────────────────────────────

def build_static() -> None:
    os.makedirs(STATIC_DIR, exist_ok=True)
    out_path = f"{STATIC_DIR}/edges_static.parquet"

    if os.path.exists(out_path):
        print(f"[skip] Static features already exist → {out_path}")
        return

    print("Loading graph …")
    G = ox.load_graphml(f"{GRAPH_DIR}/mumbai.graphml")

    print("Converting to GeoDataFrame …")
    _, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()          # (u, v, key) become columns

    # ── unique edge ID ─────────────────────────────────────────────────────────
    edges["edge_id"] = (
        edges["u"].astype(str) + "_" +
        edges["v"].astype(str) + "_" +
        edges["key"].astype(str)
    )

    # ── road type (normalised) ─────────────────────────────────────────────────
    edges["road_type"] = edges["highway"].apply(_normalise_road_type)

    # ── free-flow speed (km/h) ─────────────────────────────────────────────────
    raw_speed = (
        edges["maxspeed"].apply(_parse_maxspeed)
        if "maxspeed" in edges.columns
        else pd.Series(np.nan, index=edges.index)
    )
    edges["free_flow_speed"] = (
        raw_speed
        .fillna(edges["road_type"].map(FREE_FLOW_SPEED))
        .fillna(30.0)
        .clip(lower=5.0)
        .astype(np.float32)
    )

    # ── num lanes ─────────────────────────────────────────────────────────────
    raw_lanes = (
        edges["lanes"].apply(_parse_lanes)
        if "lanes" in edges.columns
        else pd.Series(np.nan, index=edges.index)
    )
    edges["num_lanes"] = (
        raw_lanes
        .fillna(edges["road_type"].map(LANE_DEFAULTS))
        .fillna(1.0)
        .clip(lower=1)
        .astype(np.int8)
    )

    # ── oneway ────────────────────────────────────────────────────────────────
    edges["oneway"] = (
        (edges["oneway"].apply(_parse_oneway) if "oneway" in edges.columns
         else pd.Series(False, index=edges.index))
        .astype(bool)
    )

    # ── road length (m) ───────────────────────────────────────────────────────
    edges["road_length"] = edges["length"].fillna(50.0).clip(lower=1.0).astype(np.float32)

    # ── centroid lat / lon ────────────────────────────────────────────────────
    edges["lon"] = edges.geometry.centroid.x.astype(np.float32)
    edges["lat"] = edges.geometry.centroid.y.astype(np.float32)

    # ── ordinal road type encoding ────────────────────────────────────────────
    edges["road_type_enc"] = (
        edges["road_type"].map(ROAD_TYPE_ENC).fillna(1).astype(np.int8)
    )

    # ── congestion susceptibility ─────────────────────────────────────────────
    edges["susceptibility"] = (
        edges["road_type"].map(SUSCEPTIBILITY).fillna(0.75).astype(np.float32)
    )

    # ── traffic signal count on this edge (u or v node tagged) ───────────────
    signal_nodes = {
        n for n, d in G.nodes(data=True)
        if d.get("highway") == "traffic_signals"
    }
    edges["traffic_signal_count"] = (
        edges["u"].isin(signal_nodes).astype(np.int8) +
        edges["v"].isin(signal_nodes).astype(np.int8)
    )

    # ── intersection count (degree > 2 means real junction) ──────────────────
    deg = dict(G.degree())
    edges["intersection_count"] = (
        edges["u"].map(lambda n: int(deg.get(n, 0) > 2)) +
        edges["v"].map(lambda n: int(deg.get(n, 0) > 2))
    ).astype(np.int8)

    # ── signals per km ────────────────────────────────────────────────────────
    edges["signals_per_km"] = (
        edges["traffic_signal_count"] /
        (edges["road_length"] / 1000.0).clip(lower=0.01)
    ).astype(np.float32)

    # ── zone ID   (FIELD_RES° grid, ≈2.2 km cells) ───────────────────────────
    lat_min = float(edges["lat"].min())
    lon_min = float(edges["lon"].min())
    z_lat = ((edges["lat"] - lat_min) / FIELD_RES).astype(int)
    z_lon = ((edges["lon"] - lon_min) / FIELD_RES).astype(int)
    zone_keys = list(zip(z_lat, z_lon))
    zone_map  = {k: i for i, k in enumerate(dict.fromkeys(zone_keys))}
    edges["zone_id"] = pd.array([zone_map[k] for k in zone_keys], dtype="int32")

    # ── corridor ID  (0.10° grid, ≈11 km super-zones) ────────────────────────
    c_lat = ((edges["lat"] - lat_min) / 0.10).astype(int)
    c_lon = ((edges["lon"] - lon_min) / 0.10).astype(int)
    corr_keys = list(zip(c_lat, c_lon))
    corr_map  = {k: i for i, k in enumerate(dict.fromkeys(corr_keys))}
    edges["corridor_id"] = pd.array([corr_map[k] for k in corr_keys], dtype="int16")

    # ── finalise ──────────────────────────────────────────────────────────────
    keep = [
        "edge_id", "u", "v",
        "lat", "lon",
        "road_type", "road_type_enc",
        "num_lanes", "oneway",
        "free_flow_speed", "road_length",
        "traffic_signal_count", "intersection_count", "signals_per_km",
        "susceptibility",
        "zone_id", "corridor_id",
    ]
    df = (
        edges[keep]
        .drop_duplicates(subset="edge_id")
        .reset_index(drop=True)
        .copy()
    )

    print(
        f"Edges      : {len(df):,}\n"
        f"Zones      : {df['zone_id'].nunique()}\n"
        f"Corridors  : {df['corridor_id'].nunique()}\n"
        f"Road types : {df['road_type'].value_counts().to_dict()}"
    )

    df.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}\n")


if __name__ == "__main__":
    build_static()