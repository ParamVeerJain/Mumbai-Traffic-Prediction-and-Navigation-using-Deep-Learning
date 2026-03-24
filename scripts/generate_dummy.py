# scripts/generate_dummy.py
"""
Generates a dummy road graph + dummy CSV history.
Run once before launching the UI.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.graph import load_graph, DATA_DIR
from config import DATA_DIR, MODELS_DIR

GRAPH_CACHE = os.path.join(DATA_DIR, "graph.pkl")
CSV_OUT     = os.path.join(DATA_DIR, "dummy_history.csv")
IDX_OUT     = os.path.join(DATA_DIR, "road_to_idx.pkl")


def main():
    os.makedirs(DATA_DIR,   exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Build/cache graph
    G = load_graph(cache_path=GRAPH_CACHE)
    edges = list(G.edges(data=True))
    print(f"Graph edges: {len(edges)}")

    # 2. Build road_to_idx
    road_to_idx = {(u, v): i for i, (u, v, _) in enumerate(edges)}
    with open(IDX_OUT, "wb") as f:
        pickle.dump(road_to_idx, f)
    print(f"Saved road_to_idx: {len(road_to_idx)} roads → {IDX_OUT}")

    # 3. Generate dummy history CSV (7 days × 24 hrs per edge)
    # Add per-road variation so LSTM has something to learn
    rows = []
    timestamps = pd.date_range("2024-01-01", periods=168, freq="1h")

    for i, (u, v, attr) in enumerate(edges[:500]):   # cap at 500 for speed
        phase = np.random.uniform(-3, 3)
        amp   = np.random.uniform(0.10, 0.30)
        bias  = np.random.uniform(0.65, 0.95)
        ff    = attr.get("free_flow_speed", 40.0)
        length = attr.get("length_m", 300.0)

        for ts in timestamps:
            h   = ts.hour
            dow = ts.weekday()
            sr  = bias * (0.75 + amp * np.sin(2*np.pi*(h-6+phase)/24))
            sr *= (0.85 + 0.15*np.cos(2*np.pi*dow/7))
            sr  = float(np.clip(sr + np.random.normal(0, 0.02), 0.05, 1.0))
            cong = 1.0 - sr
            speed = sr * ff
            tt    = length / (speed * 1000/3600 + 1e-6)

            rows.append({
                "u": u, "v": v,
                "timestamp": ts,
                "hour": h,
                "day_of_week": dow,
                "speed_ratio": round(sr, 4),
                "congestion":  round(cong, 4),
                "rain":        int(np.random.random() < 0.15),
                "incident":    int(np.random.random() < 0.05),
                "current_speed":   round(speed, 3),
                "free_flow_speed": round(ff, 3),
                "length_m":        round(length, 2),
                "travel_time_log_Scaled": round(float(np.log1p(tt)/6.0), 4),
                "length_log_Scaled":      round(float(np.log1p(length)/7.0), 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)
    print(f"Saved {len(df):,} rows → {CSV_OUT}")


if __name__ == "__main__":
    main()