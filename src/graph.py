# src/graph.py
"""
OSM graph loading + nearest-node snapping.
Falls back to a tiny synthetic graph when OSM is unavailable (for offline POC).
"""

import os, pickle, math
import numpy as np
import networkx as nx

try:
    import osmnx as ox
    _OSX_AVAILABLE = True
except ImportError:
    _OSX_AVAILABLE = False

from config import DATA_DIR, OSM_NETWORK_TYPE, OSM_DIST_M


# ── public API ────────────────────────────────────────────────────────────────

def load_graph(centre_point=None, cache_path=None):
    """
    Returns a networkx DiGraph with edges carrying:
      length_m, free_flow_speed, highway, traffic_signals (bool), toll (bool),
      lat/lon for each node.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[graph] Loading cached graph from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if _OSX_AVAILABLE and centre_point:
        G = _download_osm(centre_point)
    else:
        print("[graph] OSMnx unavailable or no centre_point — using synthetic graph")
        G = _synthetic_graph()

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(G, f)

    return G


def snap_to_node(G, lat, lon):
    """Return the node id nearest to (lat, lon)."""
    best_node, best_dist = None, float("inf")
    for node, data in G.nodes(data=True):
        d = _haversine(lat, lon, data["lat"], data["lon"])
        if d < best_dist:
            best_dist, best_node = d, node
    return best_node


def haversine_m(lat1, lon1, lat2, lon2):
    return _haversine(lat1, lon1, lat2, lon2)


# ── internal ──────────────────────────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _download_osm(centre_point):
    print(f"[graph] Downloading OSM graph around {centre_point} ...")
    G_raw = ox.graph_from_point(centre_point, dist=OSM_DIST_M,
                                network_type=OSM_NETWORK_TYPE)
    G_raw = ox.add_edge_speeds(G_raw)
    G_raw = ox.add_edge_travel_times(G_raw)

    G = nx.DiGraph()
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_raw)

    # nodes
    for nid, row in nodes_gdf.iterrows():
        G.add_node(nid, lat=row.geometry.y, lon=row.geometry.x)

    # edges
    for (u, v, _), row in edges_gdf.iterrows():
        highway  = row.get("highway", "residential")
        if isinstance(highway, list):
            highway = highway[0]
        speed    = float(row.get("speed_kph", 40))
        length   = float(row.get("length", 100))
        signals  = "traffic_signals" in str(row.get("highway", ""))
        toll     = "toll" in str(row.get("access", ""))
        G.add_edge(u, v,
                   length_m         = length,
                   free_flow_speed  = speed,
                   highway          = highway,
                   traffic_signals  = signals,
                   toll             = toll,
                   predicted_speed  = speed,   # will be overwritten by LSTM
                   travel_time_s    = length / (speed * 1000/3600 + 1e-6))
    print(f"[graph] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def _synthetic_graph():
    """
    A tiny 6×6 grid graph around Mumbai for offline POC.
    Nodes carry lat/lon; edges carry traffic attributes.
    """
    import random
    random.seed(42)
    np.random.seed(42)

    G    = nx.DiGraph()
    ROWS, COLS = 8, 8
    lat0, lon0 = 19.07, 72.88
    dlat, dlon = 0.008, 0.008

    highway_types = ["primary","secondary","tertiary","residential","motorway"]
    node_id = 0
    grid    = {}

    for r in range(ROWS):
        for c in range(COLS):
            nid = node_id
            grid[(r, c)] = nid
            G.add_node(nid,
                       lat = lat0 + r * dlat,
                       lon = lon0 + c * dlon)
            node_id += 1

    def _add(u, v, length, speed, hw, signals, toll):
        G.add_edge(u, v,
                   length_m        = length,
                   free_flow_speed = speed,
                   highway         = hw,
                   traffic_signals = signals,
                   toll            = toll,
                   predicted_speed = speed,
                   travel_time_s   = length / (speed * 1000/3600 + 1e-6))
        G.add_edge(v, u,
                   length_m        = length,
                   free_flow_speed = speed,
                   highway         = hw,
                   traffic_signals = signals,
                   toll            = toll,
                   predicted_speed = speed,
                   travel_time_s   = length / (speed * 1000/3600 + 1e-6))

    for r in range(ROWS):
        for c in range(COLS):
            length  = random.uniform(300, 1200)
            speed   = random.choice([30, 40, 50, 60, 80])
            hw      = random.choice(highway_types)
            signals = random.random() < 0.25
            toll    = random.random() < 0.08
            if c + 1 < COLS:
                _add(grid[(r,c)], grid[(r,c+1)], length, speed, hw, signals, toll)
            if r + 1 < ROWS:
                _add(grid[(r,c)], grid[(r+1,c)], length, speed, hw, signals, toll)

    print(f"[graph] Synthetic graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G