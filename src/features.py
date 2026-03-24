# src/features.py
"""
Build the 10-feature LSTM input sequence for a given edge at a given datetime.
For a POC, uses dummy/historical values. In production, feed live TomTom data.
"""

import numpy as np
import pandas as pd
import math
from config import FEATURE_COLS, SEQ_LEN


def build_edge_sequence(edge_data: dict, departure_dt: pd.Timestamp,
                        df_history=None) -> np.ndarray:
    """
    Returns shape (SEQ_LEN, 10) — the last SEQ_LEN hours of features
    for this edge, ending at departure_dt.

    edge_data keys: length_m, free_flow_speed, predicted_speed (current),
                    congestion (current), rain (current), incident (current)
    """
    seq = []

    for step in range(SEQ_LEN):
        # Walk backwards: step 0 = departure_dt, step 1 = 1 hr before, ...
        t = departure_dt - pd.Timedelta(hours=(SEQ_LEN - 1 - step))

        hour = t.hour
        dow  = t.weekday()

        # Cyclical time
        h_sin = math.sin(2 * math.pi * hour / 24)
        h_cos = math.cos(2 * math.pi * hour / 24)
        d_sin = math.sin(2 * math.pi * dow  / 7)
        d_cos = math.cos(2 * math.pi * dow  / 7)

        # Speed ratio at this historical step (or estimate from current)
        if df_history is not None:
            row = _lookup_history(df_history, edge_data, t)
        else:
            row = None

        if row is not None:
            sr          = float(row["speed_ratio"])
            congestion  = float(row["congestion"])
            rain        = float(row["rain"])
            incident    = float(row["incident"])
            tt_scaled   = float(row["travel_time_log_Scaled"])
            len_scaled  = float(row["length_log_Scaled"])
        else:
            # Fallback: synthesise from edge static attributes + time-of-day
            sr         = _synthetic_speed_ratio(edge_data, hour, dow)
            congestion = 1.0 - sr
            rain       = 0.0
            incident   = 0.0
            tt_scaled  = _approx_tt_scaled(edge_data)
            len_scaled = _approx_len_scaled(edge_data)

        seq.append([h_sin, h_cos, d_sin, d_cos,
                    tt_scaled, rain, incident, congestion,
                    len_scaled, sr])

    return np.array(seq, dtype=np.float32)   # (SEQ_LEN, 10)


# ── helpers ───────────────────────────────────────────────────────────────────

def _synthetic_speed_ratio(edge_data, hour, dow):
    base    = 0.75
    phase   = edge_data.get("phase_shift", 0.0)
    amp     = edge_data.get("amplitude",   0.20)
    bias    = edge_data.get("road_bias",   0.85)
    sr      = bias * (base + amp * math.sin(2*math.pi*(hour - 6 + phase)/24))
    sr     *= (0.85 + 0.15 * math.cos(2*math.pi*dow/7))
    return float(np.clip(sr, 0.05, 1.0))


def _approx_tt_scaled(edge_data):
    tt = edge_data.get("length_m", 500) / (edge_data.get("free_flow_speed", 40) * 1000/3600 + 1e-6)
    return float(np.log1p(tt) / 6.0)   # rough robust scale


def _approx_len_scaled(edge_data):
    return float(np.log1p(edge_data.get("length_m", 500)) / 7.0)


def _lookup_history(df, edge_data, t):
    u = edge_data.get("u")
    v = edge_data.get("v")
    if u is None or v is None or df is None:
        return None
    mask = ((df["u"] == u) & (df["v"] == v) &
            (df["timestamp"].dt.hour == t.hour) &
            (df["timestamp"].dt.weekday == t.weekday()))
    sub = df[mask]
    return sub.iloc[-1] if len(sub) > 0 else None