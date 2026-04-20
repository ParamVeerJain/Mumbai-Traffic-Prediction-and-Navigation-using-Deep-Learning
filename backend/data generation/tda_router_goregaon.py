import argparse
import glob
import heapq
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")

GRAPH_PATH = os.path.join(DATA_ROOT, "graph", "mumbai.graphml")
STATIC_PATH = os.path.join(DATA_ROOT, "static", "edges_static.parquet")
TS_GLOB = os.path.join(DATA_ROOT, "timeseries", "batch_*.parquet")
RAW_WEEK_START = pd.Timestamp("2024-07-01 00:00:00")

LAT_MIN, LAT_MAX = 19.1400, 19.1800
LON_MIN, LON_MAX = 72.8300, 72.8900

NAMED_LOCATIONS = [
    ("Goregaon East", 19.1550, 72.8690),
    ("Goregaon West", 19.1650, 72.8470),
    ("Aarey Colony", 19.1480, 72.8800),
    ("Film City", 19.1510, 72.8880),
    ("Oberoi Mall", 19.1735, 72.8600),
    ("Inorbit Mall Goregaon", 19.1740, 72.8360),
    ("Goregaon Railway Station", 19.1647, 72.8493),
    ("Nesco IT Park", 19.1702, 72.8553),
    ("Nirlon Knowledge Park", 19.1678, 72.8572),
    ("Royal Palms Estate", 19.1425, 72.8830),
    ("Bangur Nagar", 19.1685, 72.8320),
    ("Jawahar Nagar", 19.1625, 72.8430),
    ("Unnat Nagar", 19.1720, 72.8385),
    ("Motilal Nagar", 19.1705, 72.8410),
    ("Pandurang Wadi", 19.1590, 72.8625),
    ("Dindoshi", 19.1760, 72.8650),
    ("St Xavier's High School Goregaon", 19.1715, 72.8420),
    ("Vibgyor High School Goregaon", 19.1752, 72.8583),
    ("SRV Hospital Goregaon", 19.1680, 72.8465),
    ("Lifeline Medicare Hospital", 19.1605, 72.8480),
    ("St John's High School Goregaon", 19.1673, 72.8456),
]

DEMO_ROUTES = [
    ("Goregaon Railway Station", "Film City"),
    ("Inorbit Mall Goregaon", "Oberoi Mall"),
    ("Aarey Colony", "Nesco IT Park"),
    ("Royal Palms Estate", "Dindoshi"),
    ("Lifeline Medicare Hospital", "Nirlon Knowledge Park"),
]


@dataclass
class TDContext:
    edge_fftt: Dict[str, float]
    edge_length_m: Dict[str, float]
    node_coords: Dict[int, Tuple[float, float]]
    max_ffs_ms: float
    eid_to_local: Dict[str, int]
    raw_ttr: np.ndarray
    gwn_ttr: Optional[np.ndarray]
    gwn_idx: Optional[Dict[str, int]]

    def get_pred_ttr(self, eid: str, hour: int) -> float:
        h = int(max(0, min(hour, 167)))
        if self.gwn_ttr is not None and self.gwn_idx is not None:
            gi = self.gwn_idx.get(eid)
            if gi is not None:
                return max(1.0, float(self.gwn_ttr[gi, h]))
        li = self.eid_to_local.get(eid)
        if li is None:
            return 1.0
        return max(1.0, float(self.raw_ttr[li, min(h + 1, 167)]))

    def heuristic_sec(self, n: int, dst: int) -> float:
        return haversine_m(self.node_coords[n], self.node_coords[dst]) / self.max_ffs_ms


def in_bbox(lat: float, lon: float) -> bool:
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    r = 6_371_000.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def nearest_node(lat: float, lon: float, node_ids: List[int], node_xy: np.ndarray) -> int:
    d = node_xy - np.array([lat, lon], dtype=np.float64)
    idx = int(np.argmin(np.sum(d * d, axis=1)))
    return int(node_ids[idx])


def maybe_load_gwn_predictions(explicit_path: Optional[str]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    candidates: List[str] = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.extend(
        [
            os.path.join(DATA_ROOT, "predictions", "gwn_ttr.parquet"),
            os.path.join(DATA_ROOT, "predictions", "gwn_predictions.parquet"),
            os.path.join(DATA_ROOT, "predictions", "horizon_tplus_01.parquet"),
        ]
    )
    for path in candidates:
        if not path:
            continue
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # Accept either explicit GWN naming or generated prediction schema.
            if {"edge_id", "hour_of_week", "pred_ttr"}.issubset(df.columns):
                return df[["edge_id", "hour_of_week", "pred_ttr"]].copy(), path
            if {"edge_id", "target_timestamp", "pred_ttr"}.issubset(df.columns):
                out = df[["edge_id", "target_timestamp", "pred_ttr"]].copy()
                out["target_timestamp"] = pd.to_datetime(out["target_timestamp"])
                out["hour_of_week"] = (
                    ((out["target_timestamp"] - RAW_WEEK_START).dt.total_seconds() // 3600).astype(int).clip(0, 167)
                )
                return out[["edge_id", "hour_of_week", "pred_ttr"]], path
    return None, None


def build_raw_ttr(eids: List[str]) -> Tuple[Dict[str, int], np.ndarray]:
    eid_to_local = {eid: i for i, eid in enumerate(eids)}
    raw_ttr = np.ones((len(eids), 168), dtype=np.float32)
    ts_files = sorted(glob.glob(TS_GLOB))
    if not ts_files:
        return eid_to_local, raw_ttr
    for fp in ts_files:
        df = pd.read_parquet(fp, columns=["edge_id", "timestamp", "travel_time_ratio"])
        df = df[df["edge_id"].isin(eid_to_local)]
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour_of_week"] = (((df["timestamp"] - RAW_WEEK_START).dt.total_seconds() // 3600).astype(int)).clip(0, 167)
        grouped = df.groupby(["edge_id", "hour_of_week"], as_index=False)["travel_time_ratio"].mean()
        for row in grouped.itertuples(index=False):
            raw_ttr[eid_to_local[row.edge_id], int(row.hour_of_week)] = max(1.0, float(row.travel_time_ratio))
    return eid_to_local, raw_ttr


def build_gwn_lookup(gwn_df: Optional[pd.DataFrame], eids: List[str]) -> Tuple[Optional[np.ndarray], Optional[Dict[str, int]]]:
    if gwn_df is None:
        return None, None
    gwn_df = gwn_df[gwn_df["edge_id"].isin(eids)].copy()
    if gwn_df.empty:
        return None, None
    unique_eids = sorted(gwn_df["edge_id"].unique().tolist())
    idx = {eid: i for i, eid in enumerate(unique_eids)}
    arr = np.ones((len(unique_eids), 168), dtype=np.float32)
    for row in gwn_df.itertuples(index=False):
        arr[idx[row.edge_id], int(max(0, min(int(row.hour_of_week), 167)))] = max(1.0, float(row.pred_ttr))
    return arr, idx


def load_goregaon_graph_and_context(gwn_path: Optional[str]) -> Tuple[nx.MultiDiGraph, TDContext, Optional[str]]:
    g_all = ox.load_graphml(GRAPH_PATH)

    in_nodes = []
    node_coords: Dict[int, Tuple[float, float]] = {}
    for n, d in g_all.nodes(data=True):
        lat = float(d.get("y"))
        lon = float(d.get("x"))
        if in_bbox(lat, lon):
            in_nodes.append(int(n))
            node_coords[int(n)] = (lat, lon)

    g = nx.MultiDiGraph()
    for n in in_nodes:
        lat, lon = node_coords[n]
        g.add_node(n, y=lat, x=lon)

    for u, v, k, d in g_all.edges(keys=True, data=True):
        u_i = int(u)
        v_i = int(v)
        if u_i in node_coords and v_i in node_coords:
            g.add_edge(u_i, v_i, key=int(k), **d)

    static_df = pd.read_parquet(STATIC_PATH, columns=["edge_id", "road_length", "free_flow_speed"])
    static_df["free_flow_speed"] = static_df["free_flow_speed"].clip(lower=1.0, upper=120.0)
    static_df["free_flow_travel_time"] = static_df["road_length"] / (static_df["free_flow_speed"] * (1000.0 / 3600.0))
    static_df["free_flow_travel_time"] = static_df["free_flow_travel_time"].clip(lower=0.5)
    edge_fftt = dict(zip(static_df["edge_id"], static_df["free_flow_travel_time"]))
    edge_length_m = dict(zip(static_df["edge_id"], static_df["road_length"]))

    max_ffs_kmh = float(static_df["free_flow_speed"].max())
    max_ffs_ms = max_ffs_kmh * 1000.0 / 3600.0
    max_ffs_ms = max(max_ffs_ms, 1.0)

    bbox_eids = []
    for u, v, k in g.edges(keys=True):
        eid = f"{u}_{v}_{k}"
        if eid in edge_fftt:
            bbox_eids.append(eid)

    eid_to_local, raw_ttr = build_raw_ttr(bbox_eids)
    gwn_df, gwn_source = maybe_load_gwn_predictions(gwn_path)
    gwn_ttr, gwn_idx = build_gwn_lookup(gwn_df, bbox_eids)

    ctx = TDContext(
        edge_fftt=edge_fftt,
        edge_length_m=edge_length_m,
        node_coords=node_coords,
        max_ffs_ms=max_ffs_ms,
        eid_to_local=eid_to_local,
        raw_ttr=raw_ttr,
        gwn_ttr=gwn_ttr,
        gwn_idx=gwn_idx,
    )
    return g, ctx, gwn_source


def tda_star(
    g: nx.MultiDiGraph,
    ctx: TDContext,
    src: int,
    dst: int,
    start_hour: int = 8,
    start_elapsed_sec: float = 0.0,
) -> Tuple[Optional[List[Tuple[int, int, int]]], float]:
    pq: List[Tuple[float, int, float, int, Optional[Tuple[int, int, int]]]] = []
    counter = 0
    heapq.heappush(pq, (ctx.heuristic_sec(src, dst), counter, 0.0, src, None))

    came_from: Dict[int, Tuple[Optional[int], Optional[Tuple[int, int, int]]]] = {src: (None, None)}
    best_g: Dict[int, float] = {src: 0.0}

    while pq:
        _, _, g_cost, u, _ = heapq.heappop(pq)
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
            pred_ttr = ctx.get_pred_ttr(eid, cur_hr)
            w = pred_ttr * fftt
            ng = g_cost + w
            if ng < best_g.get(v, float("inf")):
                best_g[v] = ng
                came_from[v] = (u, (u, v, int(k)))
                counter += 1
                heapq.heappush(pq, (ng + ctx.heuristic_sec(v, dst), counter, ng, v, (u, v, int(k))))

    if dst not in came_from:
        return None, float("inf")

    path: List[Tuple[int, int, int]] = []
    node = dst
    while node != src:
        parent, edge = came_from[node]
        if parent is None or edge is None:
            return None, float("inf")
        path.append(edge)
        node = parent
    path.reverse()
    return path, best_g[dst]


def edge_stats_for_path(
    path: List[Tuple[int, int, int]],
    ctx: TDContext,
    start_hour: int,
    start_elapsed_sec: float,
) -> List[Tuple[str, float, float, float, float]]:
    rows = []
    g_cost = 0.0
    for u, v, k in path:
        eid = f"{u}_{v}_{k}"
        cur_hr = start_hour + int((g_cost + start_elapsed_sec) / 3600.0)
        fftt = float(ctx.edge_fftt[eid])
        pred_ttr = float(ctx.get_pred_ttr(eid, cur_hr))
        w = pred_ttr * fftt
        length_m = float(ctx.edge_length_m.get(eid, 0.0))
        rows.append((eid, length_m, fftt, pred_ttr, w))
        g_cost += w
    return rows


def draw_routes(
    g: nx.MultiDiGraph,
    named_nodes: Dict[str, int],
    route_results: List[Tuple[str, str, List[Tuple[int, int, int]], float]],
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))

    for u, v in g.edges():
        y1, x1 = g.nodes[u]["y"], g.nodes[u]["x"]
        y2, x2 = g.nodes[v]["y"], g.nodes[v]["x"]
        ax.plot([x1, x2], [y1, y2], color="#d3d3d3", linewidth=0.7, alpha=0.6, zorder=1)

    colors = ["#e63946", "#1d3557", "#2a9d8f", "#f4a261", "#6a4c93"]
    for i, (src_name, dst_name, path, total_sec) in enumerate(route_results):
        color = colors[i % len(colors)]
        for u, v, _ in path:
            y1, x1 = g.nodes[u]["y"], g.nodes[u]["x"]
            y2, x2 = g.nodes[v]["y"], g.nodes[v]["x"]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.2, alpha=0.95, zorder=3)

        s = named_nodes[src_name]
        d = named_nodes[dst_name]
        ax.scatter(g.nodes[s]["x"], g.nodes[s]["y"], marker="o", s=60, color=color, zorder=4)
        ax.scatter(g.nodes[d]["x"], g.nodes[d]["y"], marker="*", s=180, color=color, zorder=5)
        ax.plot([], [], color=color, linewidth=3, label=f"{src_name} -> {dst_name} ({total_sec / 60.0:.1f} min)")

    for name, node in named_nodes.items():
        ax.text(
            g.nodes[node]["x"] + 0.0003,
            g.nodes[node]["y"] + 0.0002,
            name,
            fontsize=7,
            color="#1f2937",
            zorder=6,
        )

    ax.set_title("Goregaon Time-Dependent A* Routes")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Time-Dependent A* routing for Goregaon.")
    parser.add_argument("--start_hour", type=int, default=8, help="Start hour-of-week for routing.")
    parser.add_argument("--start_elapsed_sec", type=float, default=0.0, help="Elapsed offset in seconds before start.")
    parser.add_argument(
        "--gwn_pred_path",
        type=str,
        default=None,
        help="Optional parquet path with columns: edge_id, hour_of_week, pred_ttr.",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default=os.path.join(DATA_ROOT, "pngs", "goregaon_tda_routes.png"),
        help="Path to save combined route plot.",
    )
    args = parser.parse_args()

    g, ctx, gwn_source = load_goregaon_graph_and_context(args.gwn_pred_path)
    node_ids = list(ctx.node_coords.keys())
    node_xy = np.array([ctx.node_coords[n] for n in node_ids], dtype=np.float64)

    named_nodes: Dict[str, int] = {}
    for name, lat, lon in NAMED_LOCATIONS:
        named_nodes[name] = nearest_node(lat, lon, node_ids, node_xy)

    print(f"In-bbox nodes: {g.number_of_nodes():,} | edges: {g.number_of_edges():,}")
    print(f"GWN coverage edges: {0 if ctx.gwn_idx is None else len(ctx.gwn_idx):,}")
    if gwn_source:
        print(f"GWN predictions source: {gwn_source}")
    else:
        print("GWN predictions source: not found (using raw_ttr fallback only)")
    print("-" * 96)

    route_results: List[Tuple[str, str, List[Tuple[int, int, int]], float]] = []
    for src_name, dst_name in DEMO_ROUTES:
        src = named_nodes[src_name]
        dst = named_nodes[dst_name]
        path, total_sec = tda_star(
            g=g,
            ctx=ctx,
            src=src,
            dst=dst,
            start_hour=int(args.start_hour),
            start_elapsed_sec=float(args.start_elapsed_sec),
        )
        print(f"{src_name} ({src}) -> {dst_name} ({dst})")
        if path is None:
            print("  Route: unreachable\n")
            continue
        print(f"  Edges in path: {len(path)}")
        print(f"  Total predicted travel time: {total_sec / 60.0:.2f} minutes")
        print("  Per-edge: eid | length_m | fftt_s | pred_ttr | weighted_cost_s")

        rows = edge_stats_for_path(path, ctx, int(args.start_hour), float(args.start_elapsed_sec))
        for eid, length_m, fftt, pred_ttr, w in rows:
            print(f"    {eid} | {length_m:.1f} | {fftt:.2f} | {pred_ttr:.3f} | {w:.2f}")
        print()

        route_results.append((src_name, dst_name, path, total_sec))

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    if route_results:
        draw_routes(g, named_nodes, route_results, args.out_png)
        print(f"Saved plot: {args.out_png}")
    else:
        print("No valid routes were found; plot not generated.")


if __name__ == "__main__":
    main()
