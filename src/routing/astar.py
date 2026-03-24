# src/routing/astar.py
"""
Admissible A* router — 3 modes:
  'fastest'    — minimum travel time, no restrictions
  'no_signals' — avoids roads marked with traffic_signals
  'no_toll'    — avoids toll roads

Heuristic: haversine(node, goal) / v_max  — admissible because
           no edge can be traversed faster than the network's max speed.
"""

import math
import heapq
import networkx as nx
from src.graph import haversine_m

MODE_FASTEST    = "fastest"
MODE_NO_SIGNALS = "no_signals"
MODE_NO_TOLL    = "no_toll"


def astar_route(G: nx.DiGraph, origin: int, destination: int,
                mode: str = MODE_FASTEST):
    """
    Returns (path, total_travel_time_s, edge_details) or (None, inf, []).
    path = list of node ids.
    edge_details = list of dicts with per-edge metadata.
    """
    if origin == destination:
        return [origin], 0.0, []

    # pre-compute v_max for admissible heuristic
    v_max_ms = max(
        (d.get("free_flow_speed", 40) * 1000 / 3600
         for _, _, d in G.edges(data=True)),
        default=40 * 1000 / 3600
    )

    goal_lat = G.nodes[destination]["lat"]
    goal_lon = G.nodes[destination]["lon"]

    def h(node):
        d = haversine_m(G.nodes[node]["lat"], G.nodes[node]["lon"],
                        goal_lat, goal_lon)
        return d / v_max_ms   # seconds lower-bound

    def edge_cost(u, v, data):
        if mode == MODE_NO_SIGNALS and data.get("traffic_signals", False):
            return float("inf")
        if mode == MODE_NO_TOLL and data.get("toll", False):
            return float("inf")
        return data.get("travel_time_s", 1e6)

    # open = (f, g, node, path)
    g_score = {origin: 0.0}
    open_heap = [(h(origin), 0.0, origin, [origin])]
    closed = set()

    while open_heap:
        f, g, node, path = heapq.heappop(open_heap)

        if node in closed:
            continue
        closed.add(node)

        if node == destination:
            edge_details = _collect_edge_details(G, path)
            return path, g, edge_details

        for nbr, data in G[node].items():
            if nbr in closed:
                continue
            cost = edge_cost(node, nbr, data)
            if cost == float("inf"):
                continue
            ng = g + cost
            if ng < g_score.get(nbr, float("inf")):
                g_score[nbr] = ng
                heapq.heappush(open_heap,
                               (ng + h(nbr), ng, nbr, path + [nbr]))

    return None, float("inf"), []


def _collect_edge_details(G, path):
    details = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        d    = G[u][v]
        details.append({
            "u": u, "v": v,
            "length_m":        d.get("length_m", 0),
            "free_flow_speed": d.get("free_flow_speed", 40),
            "predicted_speed": d.get("predicted_speed", 40),
            "speed_ratio":     d.get("speed_ratio", 1.0),
            "travel_time_s":   d.get("travel_time_s", 0),
            "traffic_signals": d.get("traffic_signals", False),
            "toll":            d.get("toll", False),
            "highway":         d.get("highway", "unknown"),
            "u_lat": G.nodes[u]["lat"], "u_lon": G.nodes[u]["lon"],
            "v_lat": G.nodes[v]["lat"], "v_lon": G.nodes[v]["lon"],
        })
    return details


def route_summary(edge_details):
    total_dist   = sum(e["length_m"] for e in edge_details)
    total_time_s = sum(e["travel_time_s"] for e in edge_details)
    avg_speed    = (total_dist / (total_time_s + 1e-6)) * 3.6
    n_signals    = sum(1 for e in edge_details if e["traffic_signals"])
    n_toll       = sum(1 for e in edge_details if e["toll"])
    return {
        "distance_km":   round(total_dist / 1000, 2),
        "travel_min":    round(total_time_s / 60,  1),
        "avg_speed_kmh": round(avg_speed, 1),
        "signal_roads":  n_signals,
        "toll_roads":    n_toll,
        "n_edges":       len(edge_details),
    }