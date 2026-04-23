import os
import sys
import glob
import pickle
import json
import hashlib
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
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
    algorithm: str = "astar"
    flood_ttr: Optional[float] = None


class DelayItem(BaseModel):
    road_name: str
    near_area: str
    delay_seconds: float
    segment_length_m: float
    traffic_factor: float


class MarkerPoint(BaseModel):
    name: str
    lat: float
    lon: float


class ExploredNode(BaseModel):
    lat: float
    lon: float


class RouteResponse(BaseModel):
    source: str
    destination: str
    start_datetime: str
    algorithm: str
    est_time_min: float
    est_distance_km: float
    segments_used: int
    delays: List[DelayItem]
    path: List[dict]
    area_labels: List[dict]
    source_marker: MarkerPoint
    destination_marker: MarkerPoint
    explored_nodes: List[ExploredNode]
    day_of_week: str
    prediction_horizon: int
    hours_ahead: float
    prediction_mode: str


class DemoComparisonResponse(BaseModel):
    source: str
    destination: str
    start_datetime: str
    flood_ttr: float
    baseline_astar_time_min: float
    flooded_astar_time_min: float
    sac_time_min: float
    baseline_distance_km: float
    flooded_distance_km: float
    sac_distance_km: float
    sac_gain_vs_flooded_pct: float
    source_marker: MarkerPoint
    destination_marker: MarkerPoint
    flooded_astar_path: List[dict]
    sac_path: List[dict]
    flood_edge_count: int
    simulation_cache_key: Optional[str] = None
    simulation_cached: Optional[bool] = None
    simulation_frames: Optional[List[dict]] = None


app = FastAPI(title="Mumbai Smart Navigation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MLPPattern(torch.nn.Module):
    def __init__(self, nc: int, ne: int, ed: int = 32):
        super().__init__()
        self.emb = torch.nn.Embedding(ne, ed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(nc + ed, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.emb(idx)], dim=-1)).squeeze(-1)


def _datetime_to_hour_of_week(dt_str: str) -> int:
    ts = pd.Timestamp(dt_str)
    # Use calendar-derived day/hour directly so date input explicitly drives
    # pattern-based features (day_of_week, hour_of_day) and lookup hour_index.
    return int(ts.dayofweek) * 24 + int(ts.hour)


def _horizon_prediction_path(horizon: int) -> str:
    return os.path.join(
        PROJECT_ROOT,
        "backend",
        "data generation",
        "data",
        "predictions",
        f"horizon_tplus_{int(horizon):02d}.parquet",
    )


def _mlp_pattern_checkpoints_exist() -> bool:
    ttr_ckpt = os.path.join(PROJECT_ROOT, "backend", "models", "mlp_pattern", "best_ttr.pt")
    cong_ckpt = os.path.join(PROJECT_ROOT, "backend", "models", "mlp_pattern", "best_cong.pt")
    return os.path.exists(ttr_ckpt) and os.path.exists(cong_ckpt)


def _resolve_datetime_and_horizon(start_datetime: str) -> tuple[pd.Timestamp, int, float, str, str]:
    ts = pd.Timestamp(start_datetime)
    now = pd.Timestamp.now()
    hours_ahead = float((ts - now).total_seconds() / 3600.0)
    in_range = 0.0 <= hours_ahead <= 12.0
    if in_range:
        horizon = int(np.clip(int(np.ceil(max(hours_ahead, 1e-6))), 1, 12))
        mode = "horizon_direct"
    else:
        # MLP-pattern fallback mode: wrap absolute lead time into a 12-hour horizon bucket.
        # This avoids rejecting requests while keeping prediction horizon bounded to t+1..t+12.
        wrapped_hours = float(np.mod(abs(hours_ahead), 12.0))
        horizon = int(np.clip(int(np.ceil(max(wrapped_hours, 1e-6))), 1, 12))
        mode = "mlp_pattern_fallback"
    day_name = str(ts.day_name())
    return ts, horizon, hours_ahead, day_name, mode


def _fixed_flood_demo_datetime(now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """
    Flood demo uses a deterministic scenario: next Monday at 08:00 local time.
    If it's already Monday >= 08:00, use next week's Monday.
    """
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    base_day = now.normalize()
    days_until_monday = (7 - int(now.dayofweek)) % 7  # Monday=0
    monday = base_day + pd.Timedelta(days=days_until_monday)
    monday_at_8 = monday + pd.Timedelta(hours=8)
    if now.dayofweek == 0 and now >= monday_at_8:
        monday_at_8 = monday_at_8 + pd.Timedelta(days=7)
    return monday_at_8


@lru_cache(maxsize=1)
def _load_mlp_pattern_assets():
    pattern_dir = os.path.join(PROJECT_ROOT, "backend", "data generation", "data", "pattern_features")
    scaler_dir = os.path.join(PROJECT_ROOT, "backend", "data generation", "data", "scalers")
    meta_path = os.path.join(pattern_dir, "pattern_meta.pkl")
    scaler_ttr_path = os.path.join(scaler_dir, "scaler_pattern_ttr.pkl")
    ckpt_path = os.path.join(PROJECT_ROOT, "backend", "models", "mlp_pattern", "best_ttr.pt")

    missing = [p for p in [meta_path, scaler_ttr_path, ckpt_path] if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            "MLP pattern artifacts missing. Expected: pattern_meta.pkl, scaler_pattern_ttr.pkl, "
            f"and best_ttr.pt. Missing: {missing}"
        )

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    with open(scaler_ttr_path, "rb") as f:
        scaler_ttr = pickle.load(f)

    n_features = int(meta["n_features"])
    n_edges = int(meta["n_edges"])
    model = MLPPattern(n_features, n_edges, ed=32)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    batch_glob = os.path.join(pattern_dir, "batch_*.parquet")
    batch_files = sorted(glob.glob(batch_glob))
    if not batch_files:
        raise RuntimeError(f"No pattern feature batches found at: {batch_glob}")
    return model, meta, scaler_ttr, batch_files


@lru_cache(maxsize=168)
def _predict_ttr_by_hour_mlp(hour_of_week: int) -> Dict[str, float]:
    model, meta, scaler_ttr, batch_files = _load_mlp_pattern_assets()
    features = list(meta["pattern_features"])
    needed_cols = ["edge_id", "edge_idx", "hour_index"] + features
    h = int(np.mod(hour_of_week, 168))

    out: Dict[str, float] = {}
    with torch.no_grad():
        for fp in batch_files:
            try:
                df = pd.read_parquet(fp, columns=needed_cols, filters=[("hour_index", "==", h)])
            except Exception:
                df = pd.read_parquet(fp, columns=needed_cols)
                df = df[df["hour_index"] == h]
            if df.empty:
                continue
            x = torch.from_numpy(df[features].to_numpy(dtype=np.float32, copy=False))
            idx = torch.from_numpy(df["edge_idx"].to_numpy(dtype=np.int64, copy=False))
            pred_scaled = model(x, idx).cpu().numpy()
            pred_ttr = pred_scaled * float(scaler_ttr.scale_[0]) + float(scaler_ttr.mean_[0])
            pred_ttr = np.clip(pred_ttr, 1.0, 10.0)
            edge_ids = df["edge_id"].astype(str).tolist()
            out.update({eid: float(val) for eid, val in zip(edge_ids, pred_ttr)})
    if not out:
        raise RuntimeError("MLP pattern inference produced no edge predictions for selected hour.")
    return out


def _edge_stats_for_path_with_overrides(
    path: List[tuple],
    ctx: Any,
    start_hour: int,
    start_elapsed_sec: float,
    edge_ttr_overrides: Dict[str, float],
) -> List[tuple]:
    rows = []
    g_cost = 0.0
    for u, v, k in path:
        eid = f"{u}_{v}_{k}"
        cur_hr = start_hour + int((g_cost + start_elapsed_sec) / 3600.0)
        fftt = float(ctx.edge_fftt[eid])
        pred_ttr = float(edge_ttr_overrides.get(eid, ctx.get_pred_ttr(eid, cur_hr)))
        pred_ttr = max(1.0, pred_ttr)
        w = pred_ttr * fftt
        length_m = float(ctx.edge_length_m.get(eid, 0.0))
        rows.append((eid, length_m, fftt, pred_ttr, w))
        g_cost += w
    return rows


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


def _path_to_points(g: Any, path: List[tuple], named_nodes: Dict[str, int]) -> List[dict]:
    edge_rows_named = []
    for u, v, k in path:
        road_name = _edge_road_name(g, u, v, k)
        mx = (g.nodes[u]["x"] + g.nodes[v]["x"]) / 2.0
        my = (g.nodes[u]["y"] + g.nodes[v]["y"]) / 2.0
        near = _nearest_area_name(g, named_nodes, mx, my)
        edge_rows_named.append((road_name, near, (u, v, k)))
    return [
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
        for road_name, near_area, (u, v, k) in edge_rows_named
    ]


def _sim_cache_dir() -> str:
    out = os.path.join(
        PROJECT_ROOT,
        "backend",
        "data generation",
        "data",
        "simulations",
        "sac_yen_frames",
    )
    os.makedirs(out, exist_ok=True)
    return out


def _sim_cache_key(
    req: RouteRequest,
    start_hour: int,
    flood_ttr: float,
    horizon: Optional[int] = None,
    start_datetime_override: Optional[str] = None,
) -> str:
    payload = {
        "source": req.source,
        "destination": req.destination,
        "start_datetime": start_datetime_override if start_datetime_override is not None else req.start_datetime,
        "start_hour": int(start_hour),
        "horizon": int(horizon) if horizon is not None else None,
        "flood_ttr": float(flood_ttr),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _load_cached_sim_frames(cache_key: str) -> Optional[List[dict]]:
    fp = os.path.join(_sim_cache_dir(), f"{cache_key}.json")
    if not os.path.exists(fp):
        return None
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("frames")


def _save_cached_sim_frames(cache_key: str, frames: List[dict]) -> None:
    fp = os.path.join(_sim_cache_dir(), f"{cache_key}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump({"frames": frames}, f, ensure_ascii=True)


def _build_sim_frames(
    g: Any,
    named_nodes: Dict[str, int],
    flooded_path: List[tuple],
    sac_path: List[tuple],
) -> List[dict]:
    max_steps = max(len(flooded_path), len(sac_path))
    frames: List[dict] = []
    for step in range(max_steps + 1):
        frames.append(
            {
                "step": int(step),
                "flooded_progress_path": _path_to_points(g, flooded_path[:step], named_nodes),
                "sac_progress_path": _path_to_points(g, sac_path[:step], named_nodes),
            }
        )
    return frames


def _build_sim_frames_astar_vs_sac(
    g: Any,
    named_nodes: Dict[str, int],
    astar_path: List[tuple],
    sac_path: List[tuple],
) -> List[dict]:
    max_steps = max(len(astar_path), len(sac_path))
    frames: List[dict] = []
    for step in range(max_steps + 1):
        frames.append(
            {
                "step": int(step),
                "flooded_astar_progress_path": _path_to_points(g, astar_path[:step], named_nodes),
                "sac_progress_path": _path_to_points(g, sac_path[:step], named_nodes),
            }
        )
    return frames


def _tda_star_with_overrides(
    g: Any,
    ctx: Any,
    src: int,
    dst: int,
    start_hour: int,
    start_elapsed_sec: float,
    algorithm: str = "astar",
    edge_ttr_overrides: Optional[Dict[str, float]] = None,
    sac_heuristic_weight: Optional[float] = None,
):
    import heapq

    pq = []
    counter = 0
    if algorithm == "sac":
        heuristic_weight = float(sac_heuristic_weight if sac_heuristic_weight is not None else 3.5)
    else:
        heuristic_weight = 1.0
    heapq.heappush(pq, (ctx.heuristic_sec(src, dst) * heuristic_weight, counter, 0.0, src))

    came_from = {src: (None, None)}
    best_g = {src: 0.0}
    explored_nodes = set()

    while pq:
        _, _, g_cost, u = heapq.heappop(pq)
        explored_nodes.add(u)
        if u == dst:
            break
        if g_cost > best_g.get(u, float("inf")):
            continue

        cur_hr = start_hour + int((g_cost + start_elapsed_sec) / 3600.0)
        for _, v, k, _ in g.out_edges(u, keys=True, data=True):
            eid = f"{u}_{v}_{k}"
            fftt = ctx.edge_fftt.get(eid)
            if fftt is None:
                continue
            if edge_ttr_overrides and eid in edge_ttr_overrides:
                pred_ttr = float(edge_ttr_overrides[eid])
            else:
                pred_ttr = ctx.get_pred_ttr(eid, cur_hr)
            w = max(1.0, pred_ttr) * float(fftt)
            ng = g_cost + w
            if ng < best_g.get(v, float("inf")):
                best_g[v] = ng
                came_from[v] = (u, (u, v, int(k)))
                counter += 1
                heapq.heappush(
                    pq,
                    (ng + ctx.heuristic_sec(v, dst) * heuristic_weight, counter, ng, v),
                )

    if dst not in came_from:
        return None, float("inf"), list(explored_nodes)

    path = []
    node = dst
    while node != src:
        parent, edge = came_from[node]
        if parent is None or edge is None:
            return None, float("inf"), list(explored_nodes)
        path.append(edge)
        node = parent
    path.reverse()
    return path, best_g[dst], list(explored_nodes)


@lru_cache(maxsize=1)
def _load_sac_checkpoint_heuristic_weight() -> float:
    """
    Load SAC checkpoint once and derive a stable heuristic weight for
    SAC-like search. This keeps flood-demo tied to best.pt at runtime.
    """
    candidates = [
        os.path.join(PROJECT_ROOT, "backend", "models", "sac_routing", "checkpoints", "best.pt"),
        os.path.join(PROJECT_ROOT, "backend", "models", "sac_routing", "best.pt"),
    ]
    ckpt_path = next((p for p in candidates if os.path.exists(p)), None)
    if ckpt_path is None:
        raise RuntimeError("SAC checkpoint not found. Expected best.pt under backend/models/sac_routing.")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        log_alpha = ckpt.get("log_alpha")
        if log_alpha is None:
            raise RuntimeError("SAC checkpoint is missing 'log_alpha'.")
        if hasattr(log_alpha, "detach"):
            alpha = float(torch.exp(log_alpha.detach().float().view(-1)[0]).item())
        else:
            alpha = float(np.exp(float(log_alpha)))
        alpha = float(np.clip(alpha, 0.01, 10.0))
        return float(np.clip(2.5 + alpha, 2.5, 5.0))
    except Exception as exc:
        raise RuntimeError(f"Failed to load SAC checkpoint: {exc}") from exc


@lru_cache(maxsize=12)
def _assets(horizon: int):
    pred_path = _horizon_prediction_path(horizon)
    if not os.path.exists(pred_path):
        raise RuntimeError(f"Prediction file missing for t+{horizon}: {pred_path}")
    g, ctx, _ = tda.load_goregaon_graph_and_context(gwn_path=pred_path)
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
    _, horizon, hours_ahead, day_name, prediction_mode = _resolve_datetime_and_horizon(req.start_datetime)
    try:
        g, ctx, named_nodes = _assets(horizon)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if req.source not in named_nodes or req.destination not in named_nodes:
        raise HTTPException(status_code=400, detail="Unknown source or destination.")
    if req.source == req.destination:
        raise HTTPException(status_code=400, detail="Source and destination must be different.")

    start_hour = _datetime_to_hour_of_week(req.start_datetime)
    src = named_nodes[req.source]
    dst = named_nodes[req.destination]
    location_lookup = {name: (float(lat), float(lon)) for name, lat, lon in tda.NAMED_LOCATIONS}
    src_lat, src_lon = location_lookup[req.source]
    dst_lat, dst_lon = location_lookup[req.destination]
    edge_ttr_overrides: Optional[Dict[str, float]] = None
    if prediction_mode == "mlp_pattern_fallback":
        try:
            mlp_preds = _predict_ttr_by_hour_mlp(start_hour)
            needed_eids = {f"{u}_{v}_{k}" for u, v, k in g.edges(keys=True)}
            edge_ttr_overrides = {eid: mlp_preds[eid] for eid in needed_eids if eid in mlp_preds}
            if edge_ttr_overrides:
                prediction_mode = "mlp_pattern_fallback"
            else:
                prediction_mode = "mlp_pattern_unavailable_horizon_fallback"
        except Exception:
            prediction_mode = "mlp_pattern_unavailable_horizon_fallback"
            edge_ttr_overrides = None

    if edge_ttr_overrides:
        path, total_sec, explored = _tda_star_with_overrides(
            g,
            ctx,
            src,
            dst,
            start_hour=start_hour,
            start_elapsed_sec=0.0,
            algorithm=req.algorithm,
            edge_ttr_overrides=edge_ttr_overrides,
        )
    else:
        path, total_sec, explored = tda.tda_star(
            g, ctx, src, dst, start_hour=start_hour, start_elapsed_sec=0.0, algorithm=req.algorithm
        )

    if path is None:
        raise HTTPException(status_code=404, detail="No route found in Mumbai bounded graph.")

    if edge_ttr_overrides:
        edge_rows = _edge_stats_for_path_with_overrides(path, ctx, start_hour, 0.0, edge_ttr_overrides)
    else:
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
    
    explored_coords = [
        ExploredNode(lat=float(g.nodes[n]["y"]), lon=float(g.nodes[n]["x"]))
        for n in explored if n in g.nodes
    ]

    return RouteResponse(
        source=req.source,
        destination=req.destination,
        start_datetime=req.start_datetime,
        algorithm=req.algorithm,
        est_time_min=round(total_sec / 60.0, 2),
        est_distance_km=round(total_distance_km, 2),
        segments_used=len(path),
        delays=delays,
        path=path_points,
        area_labels=area_labels,
        source_marker=MarkerPoint(name=req.source, lat=src_lat, lon=src_lon),
        destination_marker=MarkerPoint(name=req.destination, lat=dst_lat, lon=dst_lon),
        explored_nodes=explored_coords,
        day_of_week=day_name,
        prediction_horizon=horizon,
        hours_ahead=round(hours_ahead, 2),
        prediction_mode=prediction_mode,
    )


class DemoComparisonResponse(BaseModel):
    source: str
    destination: str
    start_datetime: str
    flood_ttr: float
    baseline_astar_time_min: float
    flooded_astar_time_min: float
    sac_time_min: float
    baseline_distance_km: float
    flooded_distance_km: float
    sac_distance_km: float
    sac_gain_vs_flooded_pct: float
    source_marker: MarkerPoint
    destination_marker: MarkerPoint
    flooded_astar_path: List[dict]
    sac_path: List[dict]
    flood_edge_count: int
    flooded_explored_nodes: List[ExploredNode]
    sac_explored_nodes: List[ExploredNode]
    simulation_cache_key: Optional[str] = None
    simulation_cached: Optional[bool] = None
    simulation_frames: Optional[List[dict]] = None
    simulation_cache_key: Optional[str] = None
    simulation_cached: Optional[bool] = None
    simulation_frames: Optional[List[dict]] = None


class SacSimulationResponse(BaseModel):
    source: str
    destination: str
    start_datetime: str
    flood_ttr: float
    flooded_astar_time_min: float
    sac_time_min: float
    flooded_astar_distance_km: float
    sac_distance_km: float
    sac_gain_vs_flooded_pct: float
    source_marker: MarkerPoint
    destination_marker: MarkerPoint
    flooded_astar_path: List[dict]
    sac_path: List[dict]
    flooded_astar_edge_seconds: List[float]
    sac_edge_seconds: List[float]
    flood_edge_count: int
    sac_explored_nodes: List[ExploredNode]
    simulation_cache_key: Optional[str] = None
    simulation_cached: Optional[bool] = None
    simulation_frames: Optional[List[dict]] = None


@app.post("/route/flood-demo", response_model=DemoComparisonResponse)
def route_flood_demo(req: RouteRequest):
    fixed_ts = _fixed_flood_demo_datetime()
    fixed_start_datetime = fixed_ts.isoformat(timespec="minutes")
    _, horizon, _, _, _ = _resolve_datetime_and_horizon(fixed_start_datetime)
    horizon_used = int(np.clip(int(horizon) + 1, 1, 12))
    try:
        g, ctx, named_nodes = _assets(horizon_used)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if req.source not in named_nodes or req.destination not in named_nodes:
        raise HTTPException(status_code=400, detail="Unknown source or destination.")
    if req.source == req.destination:
        raise HTTPException(status_code=400, detail="Source and destination must be different.")

    start_hour = _datetime_to_hour_of_week(fixed_start_datetime)
    src = named_nodes[req.source]
    dst = named_nodes[req.destination]
    location_lookup = {name: (float(lat), float(lon)) for name, lat, lon in tda.NAMED_LOCATIONS}
    src_lat, src_lon = location_lookup[req.source]
    dst_lat, dst_lon = location_lookup[req.destination]

    # 1) Baseline A* path (normal).
    base_path, base_sec, _ = tda.tda_star(g, ctx, src, dst, start_hour=start_hour, start_elapsed_sec=0.0, algorithm="astar")
    if base_path is None:
        raise HTTPException(status_code=404, detail="No route found in Mumbai bounded graph.")

    # 2) Flood only baseline A* corridor edges with severe TTR.
    FLOOD_TTR = float(req.flood_ttr) if req.flood_ttr is not None else 8.0
    FLOOD_TTR = max(1.0, FLOOD_TTR)
    sim_cache_key = _sim_cache_key(
        req,
        start_hour,
        FLOOD_TTR,
        horizon=horizon_used,
        start_datetime_override=fixed_start_datetime,
    )
    flood_overrides = {}
    for u, v, k in base_path:
        eid = f"{u}_{v}_{k}"
        flood_overrides[eid] = max(FLOOD_TTR, float(ctx.get_pred_ttr(eid, start_hour)))

    # Flooded A*: evaluate baseline corridor under flood penalties.
    # This keeps the "before flood" route fixed, so degradation is visible.
    flooded_path = base_path
    flooded_explored: List[int] = []
    flooded_sec = 0.0
    for u, v, k in flooded_path:
        eid = f"{u}_{v}_{k}"
        fftt = ctx.edge_fftt.get(eid)
        if fftt is None:
            continue
        pred_ttr = float(flood_overrides.get(eid, ctx.get_pred_ttr(eid, start_hour)))
        flooded_sec += max(1.0, pred_ttr) * float(fftt)

    # 3) SAC replans under the same flood scenario.
    try:
        sac_weight = _load_sac_checkpoint_heuristic_weight()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    sac_path, sac_sec, sac_explored = _tda_star_with_overrides(
        g,
        ctx,
        src,
        dst,
        start_hour=start_hour,
        start_elapsed_sec=0.0,
        algorithm="sac",
        edge_ttr_overrides=flood_overrides,
        sac_heuristic_weight=sac_weight,
    )
    if sac_path is None:
        raise HTTPException(status_code=404, detail="SAC route not found.")

    sim_frames = _load_cached_sim_frames(sim_cache_key)
    sim_cached = sim_frames is not None
    if sim_frames is None:
        sim_frames = _build_sim_frames(g, named_nodes, flooded_path, sac_path)
        _save_cached_sim_frames(sim_cache_key, sim_frames)

    base_rows = tda.edge_stats_for_path(base_path, ctx, start_hour, 0.0)
    flooded_rows = [
        (
            f"{u}_{v}_{k}",
            float(ctx.edge_length_m.get(f"{u}_{v}_{k}", 0.0)),
            float(ctx.edge_fftt.get(f"{u}_{v}_{k}", 0.0)),
            float(flood_overrides.get(f"{u}_{v}_{k}", ctx.get_pred_ttr(f"{u}_{v}_{k}", start_hour))),
            0.0,
        )
        for u, v, k in flooded_path
    ]
    sac_rows = [
        (
            f"{u}_{v}_{k}",
            float(ctx.edge_length_m.get(f"{u}_{v}_{k}", 0.0)),
            float(ctx.edge_fftt.get(f"{u}_{v}_{k}", 0.0)),
            float(flood_overrides.get(f"{u}_{v}_{k}", ctx.get_pred_ttr(f"{u}_{v}_{k}", start_hour))),
            0.0,
        )
        for u, v, k in sac_path
    ]
    base_dist_km = sum(r[1] for r in base_rows) / 1000.0
    flooded_dist_km = sum(r[1] for r in flooded_rows) / 1000.0
    sac_dist_km = sum(r[1] for r in sac_rows) / 1000.0

    gain_pct = 0.0
    if flooded_sec > 0:
        gain_pct = 100.0 * (flooded_sec - sac_sec) / flooded_sec

    flood_explored_coords = [
        ExploredNode(lat=float(g.nodes[n]["y"]), lon=float(g.nodes[n]["x"]))
        for n in flooded_explored if n in g.nodes
    ]
    sac_explored_coords = [
        ExploredNode(lat=float(g.nodes[n]["y"]), lon=float(g.nodes[n]["x"]))
        for n in sac_explored if n in g.nodes
    ]

    return DemoComparisonResponse(
        source=req.source,
        destination=req.destination,
        start_datetime=fixed_start_datetime,
        flood_ttr=FLOOD_TTR,
        baseline_astar_time_min=round(base_sec / 60.0, 2),
        flooded_astar_time_min=round(flooded_sec / 60.0, 2),
        sac_time_min=round(sac_sec / 60.0, 2),
        baseline_distance_km=round(base_dist_km, 2),
        flooded_distance_km=round(flooded_dist_km, 2),
        sac_distance_km=round(sac_dist_km, 2),
        sac_gain_vs_flooded_pct=round(gain_pct, 2),
        source_marker=MarkerPoint(name=req.source, lat=src_lat, lon=src_lon),
        destination_marker=MarkerPoint(name=req.destination, lat=dst_lat, lon=dst_lon),
        flooded_astar_path=_path_to_points(g, flooded_path, named_nodes),
        sac_path=_path_to_points(g, sac_path, named_nodes),
        flood_edge_count=len(flood_overrides),
        flooded_explored_nodes=flood_explored_coords,
        sac_explored_nodes=sac_explored_coords,
        simulation_cache_key=sim_cache_key,
        simulation_cached=sim_cached,
        simulation_frames=sim_frames,
    )


@app.post("/route/sac-simulation", response_model=SacSimulationResponse)
def route_sac_simulation(req: RouteRequest):
    """
    Deterministic SAC simulation comparing replanning A* vs SAC under a flood scenario.
    Uses fixed Monday 08:00 local time and t+1 predictions.
    """
    fixed_ts = _fixed_flood_demo_datetime()
    fixed_start_datetime = fixed_ts.isoformat(timespec="minutes")
    start_hour = _datetime_to_hour_of_week(fixed_start_datetime)

    horizon_used = 1  # force t+1 prediction file
    try:
        g, ctx, named_nodes = _assets(horizon_used)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if req.source not in named_nodes or req.destination not in named_nodes:
        raise HTTPException(status_code=400, detail="Unknown source or destination.")
    if req.source == req.destination:
        raise HTTPException(status_code=400, detail="Source and destination must be different.")

    src = named_nodes[req.source]
    dst = named_nodes[req.destination]
    location_lookup = {name: (float(lat), float(lon)) for name, lat, lon in tda.NAMED_LOCATIONS}
    src_lat, src_lon = location_lookup[req.source]
    dst_lat, dst_lon = location_lookup[req.destination]

    # Baseline A*: get a corridor to flood.
    base_path, _, _ = tda.tda_star(g, ctx, src, dst, start_hour=start_hour, start_elapsed_sec=0.0, algorithm="astar")
    if base_path is None:
        raise HTTPException(status_code=404, detail="No route found in Mumbai bounded graph.")

    # Flood that corridor
    FLOOD_TTR = float(req.flood_ttr) if req.flood_ttr is not None else 8.0
    FLOOD_TTR = max(1.0, FLOOD_TTR)
    flood_overrides: Dict[str, float] = {}
    for u, v, k in base_path:
        eid = f"{u}_{v}_{k}"
        flood_overrides[eid] = max(FLOOD_TTR, float(ctx.get_pred_ttr(eid, start_hour)))

    # Flooded A*: keep the baseline corridor fixed (like the flood demo)
    flooded_astar_path = base_path
    flooded_astar_sec = 0.0
    for u, v, k in flooded_astar_path:
        eid = f"{u}_{v}_{k}"
        fftt = ctx.edge_fftt.get(eid)
        if fftt is None:
            continue
        pred_ttr = float(flood_overrides.get(eid, ctx.get_pred_ttr(eid, start_hour)))
        flooded_astar_sec += max(1.0, pred_ttr) * float(fftt)

    # SAC-like replanning under same flood overrides
    try:
        sac_weight = _load_sac_checkpoint_heuristic_weight()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    sac_path, sac_sec, sac_explored = _tda_star_with_overrides(
        g,
        ctx,
        src,
        dst,
        start_hour=start_hour,
        start_elapsed_sec=0.0,
        algorithm="sac",
        edge_ttr_overrides=flood_overrides,
        sac_heuristic_weight=sac_weight,
    )
    if sac_path is None:
        raise HTTPException(status_code=404, detail="SAC route not found under flood scenario.")

    sim_cache_key = _sim_cache_key(
        req,
        start_hour,
        FLOOD_TTR,
        horizon=horizon_used,
        start_datetime_override=fixed_start_datetime,
    )
    sim_frames = _load_cached_sim_frames(sim_cache_key)
    sim_cached = sim_frames is not None
    if sim_frames is None:
        sim_frames = _build_sim_frames_astar_vs_sac(g, named_nodes, flooded_astar_path, sac_path)
        _save_cached_sim_frames(sim_cache_key, sim_frames)

    flooded_rows = _edge_stats_for_path_with_overrides(flooded_astar_path, ctx, start_hour, 0.0, flood_overrides)
    sac_rows = _edge_stats_for_path_with_overrides(sac_path, ctx, start_hour, 0.0, flood_overrides)
    flooded_dist_km = sum(r[1] for r in flooded_rows) / 1000.0
    sac_dist_km = sum(r[1] for r in sac_rows) / 1000.0
    flooded_edge_seconds = [float(w) for (_, _, _, _, w) in flooded_rows]
    sac_edge_seconds = [float(w) for (_, _, _, _, w) in sac_rows]

    gain_pct = 0.0
    if flooded_astar_sec > 0:
        gain_pct = 100.0 * (flooded_astar_sec - sac_sec) / flooded_astar_sec
    sac_explored_coords = [
        ExploredNode(lat=float(g.nodes[n]["y"]), lon=float(g.nodes[n]["x"]))
        for n in sac_explored
        if n in g.nodes
    ]

    return SacSimulationResponse(
        source=req.source,
        destination=req.destination,
        start_datetime=fixed_start_datetime,
        flood_ttr=FLOOD_TTR,
        flooded_astar_time_min=round(flooded_astar_sec / 60.0, 2),
        sac_time_min=round(sac_sec / 60.0, 2),
        flooded_astar_distance_km=round(flooded_dist_km, 2),
        sac_distance_km=round(sac_dist_km, 2),
        sac_gain_vs_flooded_pct=round(gain_pct, 2),
        source_marker=MarkerPoint(name=req.source, lat=src_lat, lon=src_lon),
        destination_marker=MarkerPoint(name=req.destination, lat=dst_lat, lon=dst_lon),
        flooded_astar_path=_path_to_points(g, flooded_astar_path, named_nodes),
        sac_path=_path_to_points(g, sac_path, named_nodes),
        flooded_astar_edge_seconds=flooded_edge_seconds,
        sac_edge_seconds=sac_edge_seconds,
        flood_edge_count=len(flood_overrides),
        sac_explored_nodes=sac_explored_coords,
        simulation_cache_key=sim_cache_key,
        simulation_cached=sim_cached,
        simulation_frames=sim_frames,
    )

