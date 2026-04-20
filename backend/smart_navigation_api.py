import os
import sys
from functools import lru_cache
from typing import Any, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_GEN_DIR = os.path.join(PROJECT_ROOT, "backend", "data generation")
if DATA_GEN_DIR not in sys.path:
    sys.path.append(DATA_GEN_DIR)

import tda_router_goregaon as tda


class RouteRequest(BaseModel):
    source: str
    destination: str
    start_datetime: str


class DelayItem(BaseModel):
    road_name: str
    near_area: str
    delay_seconds: float
    segment_length_m: float
    traffic_factor: float


class RouteResponse(BaseModel):
    source: str
    destination: str
    start_datetime: str
    est_time_min: float
    est_distance_km: float
    segments_used: int
    delays: List[DelayItem]
    path: List[dict]
    area_labels: List[dict]


app = FastAPI(title="Mumbai Smart Navigation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _datetime_to_hour_of_week(dt_str: str) -> int:
    anchor = pd.Timestamp("2024-07-01 00:00:00")
    ts = pd.Timestamp(dt_str)
    return int(((ts - anchor).total_seconds() // 3600) % 168)


def _edge_road_name(g, u, v, k) -> str:
    data = g.get_edge_data(u, v, k) or {}
    name = data.get("name")
    if isinstance(name, list):
        name = ", ".join([str(x) for x in name if x])
    if not name:
        return "Unnamed road"
    return str(name)


def _nearest_area_name(g, named_nodes, x, y) -> str:
    nearest_name = "Nearby area"
    best_d2 = float("inf")
    for name, node_id in named_nodes.items():
        nx = g.nodes[node_id]["x"]
        ny = g.nodes[node_id]["y"]
        d2 = (nx - x) ** 2 + (ny - y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            nearest_name = name
    return nearest_name


def _edge_geometry_latlon(g: Any, u: int, v: int, k: int) -> List[List[float]]:
    data = g.get_edge_data(u, v, k) or {}
    geom = data.get("geometry")
    if geom is not None and hasattr(geom, "coords"):
        coords = list(geom.coords)
        if coords:
            return [[float(lat), float(lon)] for lon, lat in coords]
    return [
        [float(g.nodes[u]["y"]), float(g.nodes[u]["x"])],
        [float(g.nodes[v]["y"]), float(g.nodes[v]["x"])],
    ]


@lru_cache(maxsize=1)
def _assets():
    g, ctx, _ = tda.load_goregaon_graph_and_context(gwn_path=None)
    node_ids = list(ctx.node_coords.keys())
    node_xy = np.array([ctx.node_coords[n] for n in node_ids], dtype=np.float64)
    named_nodes = {}
    for name, lat, lon in tda.NAMED_LOCATIONS:
        named_nodes[name] = tda.nearest_node(lat, lon, node_ids, node_xy)
    return g, ctx, named_nodes


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/locations")
def locations():
    return {"locations": [x[0] for x in tda.NAMED_LOCATIONS]}


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest):
    g, ctx, named_nodes = _assets()
    if req.source not in named_nodes or req.destination not in named_nodes:
        raise HTTPException(status_code=400, detail="Unknown source or destination.")
    if req.source == req.destination:
        raise HTTPException(status_code=400, detail="Source and destination must be different.")

    start_hour = _datetime_to_hour_of_week(req.start_datetime)
    src = named_nodes[req.source]
    dst = named_nodes[req.destination]
    path, total_sec = tda.tda_star(g, ctx, src, dst, start_hour=start_hour, start_elapsed_sec=0.0)
    if path is None:
        raise HTTPException(status_code=404, detail="No route found in Goregaon bounded graph.")

    edge_rows = tda.edge_stats_for_path(path, ctx, start_hour, 0.0)
    total_distance_km = sum(r[1] for r in edge_rows) / 1000.0
    edge_rows_named = []
    for (u, v, k), row in zip(path, edge_rows):
        mx = (g.nodes[u]["x"] + g.nodes[v]["x"]) / 2.0
        my = (g.nodes[u]["y"] + g.nodes[v]["y"]) / 2.0
        near = _nearest_area_name(g, named_nodes, mx, my)
        edge_rows_named.append((_edge_road_name(g, u, v, k), near, row, (u, v, k)))

    top_edges = sorted(edge_rows_named, key=lambda x: x[2][4], reverse=True)[:6]
    delays = [
        DelayItem(
            road_name=road_name,
            near_area=near_area,
            delay_seconds=float(w),
            segment_length_m=float(length_m),
            traffic_factor=float(ttr),
        )
        for road_name, near_area, (_, length_m, _, ttr, w), _ in top_edges
    ]

    path_points = []
    for road_name, near_area, _, (u, v, k) in edge_rows_named:
        path_points.append(
            {
                "u_lat": float(g.nodes[u]["y"]),
                "u_lon": float(g.nodes[u]["x"]),
                "v_lat": float(g.nodes[v]["y"]),
                "v_lon": float(g.nodes[v]["x"]),
                "geometry": _edge_geometry_latlon(g, u, v, k),
                "road_name": road_name,
                "near_area": near_area,
                "edge_key": int(k),
            }
        )

    area_labels = [
        {"name": name, "lat": float(g.nodes[n]["y"]), "lon": float(g.nodes[n]["x"])}
        for name, n in named_nodes.items()
    ]

    return RouteResponse(
        source=req.source,
        destination=req.destination,
        start_datetime=req.start_datetime,
        est_time_min=round(total_sec / 60.0, 2),
        est_distance_km=round(total_distance_km, 2),
        segments_used=len(path),
        delays=delays,
        path=path_points,
        area_labels=area_labels,
    )

