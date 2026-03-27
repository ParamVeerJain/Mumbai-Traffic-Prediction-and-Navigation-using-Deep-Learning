"""
fetch_graph.py
──────────────
Downloads the OSM road network for every (lat, lon) in the config grid
using a RADIUS_M-metre circle, then composes all sub-graphs into one
unified graph and saves it as GraphML.

First run   : 10-25 min (54 Overpass API calls, network-dependent).
Repeat runs : ~5 s  (OSMnx's on-disk cache is used automatically).

Run:
    python fetch_graph.py
"""

import os
import networkx as nx
import osmnx as ox
from config import LATS, LONS, RADIUS_M, NETWORK_TYPE, GRAPH_DIR


def fetch_and_save() -> None:
    os.makedirs(GRAPH_DIR, exist_ok=True)
    out_path = f"{GRAPH_DIR}/mumbai.graphml"

    if os.path.exists(out_path):
        print(f"[skip] Graph already exists → {out_path}")
        return

    # OSMnx settings
    ox.settings.use_cache        = True
    ox.settings.log_console      = False
    ox.settings.requests_timeout = 300          # generous timeout per tile

    n_points = len(LATS) * len(LONS)
    graphs, failed = [], []

    print(f"Fetching {n_points} sub-graphs  (radius = {RADIUS_M} m, type = {NETWORK_TYPE})")
    print("Tip: if a point fails, it is skipped and the rest continue.\n")

    for i, lat in enumerate(LATS):
        for j, lon in enumerate(LONS):
            idx = i * len(LONS) + j + 1
            print(f"  [{idx:02d}/{n_points}]  ({lat:.2f}, {lon:.2f})", end="  ", flush=True)
            try:
                g = ox.graph_from_point(
                    (lat, lon),
                    dist=RADIUS_M,
                    network_type=NETWORK_TYPE,
                    simplify=True,
                    retain_all=False,
                )
                graphs.append(g)
                print(f"✓  nodes={g.number_of_nodes():,}  edges={g.number_of_edges():,}")
            except Exception as exc:
                print(f"✗  ({exc})")
                failed.append((lat, lon))

    if not graphs:
        raise RuntimeError(
            "No sub-graphs fetched. Check internet / Overpass availability."
        )

    # ── compose_all  (OSMnx deduplicates overlapping nodes/edges) ────────────
    print(f"\nComposing {len(graphs)} sub-graphs …")
    G = nx.compose_all(graphs)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"Composed graph : {n_nodes:,} nodes,  {n_edges:,} edges")

    if failed:
        print(f"Warning : {len(failed)} points failed — {failed}")

    print(f"Saving → {out_path} …")
    ox.save_graphml(G, out_path)
    print("Done.\n")


if __name__ == "__main__":
    fetch_and_save()