import argparse
import json
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from map_utils import (
    REGIONS,
    attach_geometry,
    filter_region,
    iter_line_segments,
    preprocess_timeseries_for_maps,
    toll_like_edges,
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")


def _pred_file(horizon: int) -> str:
    return os.path.join(DATA_ROOT, "predictions", f"horizon_tplus_{horizon:02d}.parquet")


def _png_root() -> str:
    root = os.path.join(DATA_ROOT, "pngs")
    os.makedirs(root, exist_ok=True)
    return root


def _frame_paths(region: str, horizon: int, target_ts: pd.Timestamp) -> Tuple[str, str]:
    stamp = target_ts.strftime("%Y%m%d_%H%M")
    base = os.path.join(_png_root(), f"tplus_{horizon:02d}", region)
    actual_dir = os.path.join(base, "actual")
    pred_dir = os.path.join(base, "pred")
    os.makedirs(actual_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    return (
        os.path.join(actual_dir, f"{stamp}.png"),
        os.path.join(pred_dir, f"{stamp}.png"),
    )


def _render_roads(
    df: pd.DataFrame,
    congestion_col: str,
    rain_col: str,
    closure_col: str,
    incident_col: str,
    region_name: str,
    out_png: str,
) -> Dict[str, float]:
    region = REGIONS[region_name]
    segs = list(iter_line_segments(df["geometry_wkt"].values))
    colors = df[congestion_col].values.astype(np.float32)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=180)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))
    ax.axis("off")
    ax.set_xlim(region.lon_min, region.lon_max)
    ax.set_ylim(region.lat_min, region.lat_max)

    lc = LineCollection(segs, cmap="coolwarm", linewidths=1.3, alpha=0.95)
    lc.set_array(colors)
    lc.set_clim(0.0, 0.97)
    ax.add_collection(lc)

    rain_mask = df[rain_col].values > 10.0
    if rain_mask.any():
        ax.scatter(df.loc[rain_mask, "lon"], df.loc[rain_mask, "lat"], s=10, marker="o", c="#5bc0ff", alpha=0.75)

    acc_mask = df[incident_col].isin([1]).values
    if acc_mask.any():
        ax.scatter(df.loc[acc_mask, "lon"], df.loc[acc_mask, "lat"], s=20, marker="X", c="#ffb000", alpha=0.85)

    jam_mask = df[incident_col].isin([6]).values
    if jam_mask.any():
        ax.scatter(df.loc[jam_mask, "lon"], df.loc[jam_mask, "lat"], s=10, marker="s", c="#ff2e2e", alpha=0.65)

    closure_mask = df[closure_col].values > 0
    if closure_mask.any():
        ax.scatter(df.loc[closure_mask, "lon"], df.loc[closure_mask, "lat"], s=20, marker="x", c="#000000", alpha=0.8)

    toll_df = toll_like_edges(region_name)
    if not toll_df.empty:
        ax.scatter(toll_df["lon"], toll_df["lat"], s=8, marker="^", c="#ffd447", alpha=0.6)

    plt.savefig(out_png, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return {
        "congestion_mean": float(np.mean(colors)),
        "congestion_max": float(np.max(colors)),
        "rain_edges": int(rain_mask.sum()),
        "accident_edges": int(acc_mask.sum()),
        "jam_edges": int(jam_mask.sum()),
        "closure_edges": int(closure_mask.sum()),
    }


def _build_actual_index(ts_df: pd.DataFrame) -> pd.DataFrame:
    return ts_df.rename(
        columns={
            "timestamp": "target_timestamp",
            "congestion_level": "actual_congestion",
            "travel_time_ratio": "actual_ttr",
            "incident_type": "actual_incident_type",
            "road_closure": "actual_road_closure",
            "hourly_rainfall_mm": "actual_rain_mm",
        }
    )


def generate(horizon: int) -> None:
    pred_path = _pred_file(horizon)
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}. Run generate_predictions.py first.")

    ts_df = preprocess_timeseries_for_maps()
    actual_df = _build_actual_index(ts_df)
    pred_df = pd.read_parquet(pred_path)

    merged = pred_df.merge(
        actual_df[
            [
                "edge_id",
                "target_timestamp",
                "actual_congestion",
                "actual_ttr",
                "actual_incident_type",
                "actual_road_closure",
                "actual_rain_mm",
            ]
        ],
        on=["edge_id", "target_timestamp"],
        how="left",
        suffixes=("", "_dup"),
    )
    for col in list(merged.columns):
        if col.endswith("_dup"):
            merged.drop(columns=[col], inplace=True)

    merged = attach_geometry(merged)
    merged["target_timestamp"] = pd.to_datetime(merged["target_timestamp"])

    week_times = sorted(merged["target_timestamp"].unique().tolist())
    week_times = [t for t in week_times if pd.Timestamp(t).hour % horizon == 0]

    for region in ("mumbai", "goregaon"):
        reg_df = filter_region(merged, region)
        meta = []
        for ts in week_times:
            frame = reg_df[reg_df["target_timestamp"] == ts].copy()
            if frame.empty:
                continue
            actual_png, pred_png = _frame_paths(region, horizon, pd.Timestamp(ts))
            if not os.path.exists(actual_png):
                stats_actual = _render_roads(
                    frame,
                    "actual_congestion",
                    "actual_rain_mm",
                    "actual_road_closure",
                    "actual_incident_type",
                    region,
                    actual_png,
                )
            else:
                stats_actual = {}
            if not os.path.exists(pred_png):
                frame["pred_incident_type"] = np.where(frame["pred_congestion"] > 0.75, 6, 0).astype(np.int8)
                frame["pred_road_closure"] = np.where(frame["pred_congestion"] > 0.92, 1, 0).astype(np.int8)
                frame["pred_rain_mm"] = frame["actual_rain_mm"].fillna(0.0)
                stats_pred = _render_roads(
                    frame,
                    "pred_congestion",
                    "pred_rain_mm",
                    "pred_road_closure",
                    "pred_incident_type",
                    region,
                    pred_png,
                )
            else:
                stats_pred = {}
            meta.append(
                {
                    "timestamp": pd.Timestamp(ts).isoformat(),
                    "actual_png": actual_png.replace("\\", "/"),
                    "pred_png": pred_png.replace("\\", "/"),
                    "actual_stats": stats_actual,
                    "pred_stats": stats_pred,
                }
            )

        meta_path = os.path.join(_png_root(), f"tplus_{horizon:02d}", region, "frames.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved {region} metadata: {meta_path} frames={len(meta)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate transparent road congestion PNG sequences.")
    parser.add_argument("--horizon", type=int, required=True, help="Prediction horizon 1..12")
    args = parser.parse_args()
    h = int(args.horizon)
    if h < 1 or h > 12:
        raise ValueError("--horizon must be between 1 and 12")
    generate(h)


if __name__ == "__main__":
    main()
