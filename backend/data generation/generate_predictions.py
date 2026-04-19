import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from map_utils import preprocess_timeseries_for_maps

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")


def _ensure_dirs() -> str:
    out_dir = os.path.join(DATA_ROOT, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _predict_temporal(
    cur_cong: np.ndarray,
    cur_ttr: np.ndarray,
    rain: np.ndarray,
    closure: np.ndarray,
    incident: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    rain_factor = np.clip(rain / 35.0, 0.0, 1.0)
    inc_factor = np.where(np.isin(incident, [1, 6, 8, 11]), 1.0, 0.0)
    horizon_scale = horizon / 12.0

    cong = cur_cong + (0.18 * horizon_scale * cur_cong) + (0.22 * rain_factor) + (0.08 * inc_factor)
    cong = np.where(closure > 0, 0.97, cong)
    cong = np.clip(cong, 0.0, 0.97)

    ttr = cur_ttr * (1.0 + 0.45 * horizon_scale * cong + 0.15 * rain_factor + 0.08 * inc_factor)
    ttr = np.where(closure > 0, np.maximum(ttr, 7.0), ttr)
    ttr = np.clip(ttr, 1.0, 10.0)
    return cong.astype(np.float32), ttr.astype(np.float32)


def _build_horizon(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    g = df.sort_values(["edge_id", "timestamp"]).copy()
    g["target_timestamp"] = g.groupby("edge_id")["timestamp"].shift(-horizon)
    g["actual_congestion"] = g.groupby("edge_id")["congestion_level"].shift(-horizon)
    g["actual_ttr"] = g.groupby("edge_id")["travel_time_ratio"].shift(-horizon)
    g["actual_incident_type"] = g.groupby("edge_id")["incident_type"].shift(-horizon)
    g["actual_road_closure"] = g.groupby("edge_id")["road_closure"].shift(-horizon)
    g["actual_rain_mm"] = g.groupby("edge_id")["hourly_rainfall_mm"].shift(-horizon)

    g = g.dropna(subset=["target_timestamp", "actual_congestion", "actual_ttr"]).copy()
    pred_cong, pred_ttr = _predict_temporal(
        g["congestion_level"].values.astype(np.float32),
        g["travel_time_ratio"].values.astype(np.float32),
        g["hourly_rainfall_mm"].values.astype(np.float32),
        g["road_closure"].values.astype(np.int8),
        g["incident_type"].values.astype(np.int8),
        horizon=horizon,
    )
    g["pred_congestion"] = pred_cong
    g["pred_ttr"] = pred_ttr

    out_cols = [
        "edge_id",
        "timestamp",
        "target_timestamp",
        "actual_congestion",
        "actual_ttr",
        "actual_incident_type",
        "actual_road_closure",
        "actual_rain_mm",
        "pred_congestion",
        "pred_ttr",
    ]
    return g[out_cols].rename(columns={"timestamp": "base_timestamp"}).reset_index(drop=True)


def generate(horizons: List[int]) -> None:
    out_dir = _ensure_dirs()
    ts = preprocess_timeseries_for_maps()
    ts = ts.sort_values(["edge_id", "timestamp"]).reset_index(drop=True)

    for h in horizons:
        out = _build_horizon(ts, h)
        path = os.path.join(out_dir, f"horizon_tplus_{h:02d}.parquet")
        out.to_parquet(path, index=False)
        print(f"Saved horizon t+{h}: {path} rows={len(out):,}")

    manifest = pd.DataFrame(
        [{"horizon": h, "file": f"horizon_tplus_{h:02d}.parquet"} for h in horizons]
    )
    manifest.to_parquet(os.path.join(out_dir, "manifest.parquet"), index=False)
    print("Saved data/predictions/manifest.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate t+1..t+12 temporal predictions for frontend.")
    parser.add_argument("--min_h", type=int, default=1)
    parser.add_argument("--max_h", type=int, default=12)
    args = parser.parse_args()

    min_h = max(1, int(args.min_h))
    max_h = min(12, int(args.max_h))
    if min_h > max_h:
        raise ValueError("--min_h must be <= --max_h")
    generate(list(range(min_h, max_h + 1)))


if __name__ == "__main__":
    main()
