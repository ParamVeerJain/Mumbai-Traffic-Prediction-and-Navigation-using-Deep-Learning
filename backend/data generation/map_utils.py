import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import osmnx as ox
from shapely.geometry import LineString

from config import GRAPH_DIR, STATIC_DIR, TIMESERIES_DIR

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")


@dataclass(frozen=True)
class Region:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


REGIONS: Dict[str, Region] = {
    "mumbai": Region("mumbai", 18.85, 19.45, 72.75, 73.05),
    "goregaon": Region("goregaon", 19.13, 19.20, 72.82, 72.88),
}


def _cache_dir() -> str:
    path = os.path.join(DATA_ROOT, "cache")
    os.makedirs(path, exist_ok=True)
    return path


@lru_cache(maxsize=1)
def load_static() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(BASE_DIR, STATIC_DIR, "edges_static.parquet"))


@lru_cache(maxsize=1)
def load_graph_edges() -> pd.DataFrame:
    cache_path = os.path.join(_cache_dir(), "edges_geometry.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    graph_path = os.path.join(GRAPH_DIR, "mumbai.graphml")
    graph_path = os.path.join(BASE_DIR, graph_path)
    G = ox.load_graphml(graph_path)
    _, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()[["u", "v", "key", "geometry"]].copy()
    edges["edge_id"] = edges["u"].astype(str) + "_" + edges["v"].astype(str) + "_" + edges["key"].astype(str)
    edges["geometry_wkt"] = edges["geometry"].astype(str)
    out = edges[["edge_id", "geometry_wkt"]]
    out.to_parquet(cache_path, index=False)
    return out


def _parse_linestring(wkt_text: str) -> LineString:
    coords_text = wkt_text.replace("LINESTRING (", "").replace(")", "")
    parts = [p.strip() for p in coords_text.split(",") if p.strip()]
    coords = []
    for part in parts:
        lon, lat = part.split(" ")
        coords.append((float(lon), float(lat)))
    return LineString(coords)


def preprocess_timeseries_for_maps() -> pd.DataFrame:
    cache_path = os.path.join(_cache_dir(), "timeseries_compact.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    files = sorted(
        f for f in os.listdir(os.path.join(BASE_DIR, TIMESERIES_DIR))
        if f.startswith("batch_") and f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError("No timeseries parquet files found in data/timeseries.")

    keep_cols = [
        "edge_id",
        "timestamp",
        "congestion_level",
        "travel_time_ratio",
        "incident_type",
        "road_closure",
        "hourly_rainfall_mm",
    ]
    frames = []
    for name in files:
        part = pd.read_parquet(os.path.join(BASE_DIR, TIMESERIES_DIR, name), columns=keep_cols)
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.to_parquet(cache_path, index=False)
    return df


def attach_geometry(df: pd.DataFrame) -> pd.DataFrame:
    geo = load_graph_edges()
    out = df.merge(geo, on="edge_id", how="left")
    missing = out["geometry_wkt"].isna().sum()
    if missing:
        raise ValueError(f"Missing geometry for {missing} edges.")
    return out


def filter_region(df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    region = REGIONS[region_name]
    static_df = load_static()[["edge_id", "lat", "lon", "road_type"]]
    static_reg = static_df[
        (static_df["lat"] >= region.lat_min)
        & (static_df["lat"] <= region.lat_max)
        & (static_df["lon"] >= region.lon_min)
        & (static_df["lon"] <= region.lon_max)
    ]
    keep_ids = set(static_reg["edge_id"].tolist())
    out = df[df["edge_id"].isin(keep_ids)].copy()
    out = out.merge(static_reg, on="edge_id", how="left")
    return out


def toll_like_edges(region_name: str) -> pd.DataFrame:
    region = REGIONS[region_name]
    static_df = load_static()[["edge_id", "lat", "lon", "road_type"]].copy()
    reg = static_df[
        (static_df["lat"] >= region.lat_min)
        & (static_df["lat"] <= region.lat_max)
        & (static_df["lon"] >= region.lon_min)
        & (static_df["lon"] <= region.lon_max)
    ]
    return reg[reg["road_type"].isin(["motorway", "motorway_link", "trunk", "trunk_link"])].copy()


def iter_line_segments(wkt_series: Iterable[str]) -> Iterable[np.ndarray]:
    for txt in wkt_series:
        line = _parse_linestring(txt)
        yield np.asarray(line.coords, dtype=np.float32)
