"""
generate_timeseries.py
───────────────────────
Generates 168 hours (1 week) of synthetic, spatially-temporally coherent
traffic data for every road edge and writes batches of BATCH_SIZE edges
to  data/timeseries/batch_XXXX.parquet.

HOW SPATIAL-TEMPORAL COHERENCE IS ACHIEVED
───────────────────────────────────────────
1.  CONGESTION FIELD  [168, n_lat, n_lon]
    A regular 0.02° grid (≈2.2 km) is laid over the study area.
    Each cell generates a time series via AR(1):

        cong[t, i, j] = α·cong[t-1, i, j]  +  (1-α)·base[t]·urban[i,j]  +  ε

    where:
      • base[t]       — Mumbai-specific hourly congestion intensity
                        (weekday: sharp 8-10am / 5-8pm peaks;
                         Saturday: gentler mid-morning; Sunday: lightest)
      • urban[i,j]    — exponential decay from Churchgate (city centre)
                        so South Mumbai roads are busiest, Vasai suburbs least
      • ε             — N(0, 0.035) iid noise
      • α = 0.68      — strong hour-to-hour autocorrelation

    A 2-D Gaussian blur (σ=1.3 cells ≈ 3 km) is applied at each timestep
    so neighbouring grid cells have correlated congestion.

2.  EVENT LAYERS  add physically-motivated deltas on top:
      • Monsoon rainfall (mm/hr) — July week, heavier on Mon/Wed/Fri
        - mild rain  → +0.04–0.12 congestion everywhere
        - heavy rain → +0.12–0.25, with extra +0.10–0.20 for coastal zones
        - flooding   → incident types 4 (Rain) / 11 (Flooding)
      • Local train disruption (Wed AM, Fri PM)
        - pushes road traffic up +0.10 for all edges (people switch to road)
      • Public holidays / school holidays (none this specific week)
        - would suppress morning peak by ~50%

3.  EDGE-LEVEL CONGESTION
    Each edge samples the field at its centroid lat/lon (nearest cell).
    Its road_type susceptibility scales the zone value:
      motorway × 0.78  (buffered — hard to fully gridlock)
      primary   × 1.05  (most affected)
      secondary × 1.10  (narrow, slow to clear)
      residential × 0.70 (low baseline)

4.  TomTom-COMPATIBLE FEATURES are derived from edge congestion:
      current_speed      = free_flow_speed × (1 − cong) + N(0, 2%)
      current_travel_time = road_length / current_speed
      travel_time_ratio  = current_tt / free_flow_tt
      confidence         ∝ speed_ratio  (sensor confidence is higher on clear roads)
      road_closure       → triggered when cong > 0.95 or rare random event

5.  INCIDENTS are sampled with probability ∝ congestion + rain + disruption.
    Types follow TomTom iconCategory; severity follows magnitudeOfDelay.
    Incident probability and type distribution shift with conditions:
      heavy rain  → more Flooding (11) and Rain (4)
      peak hour   → more Jams (6) and Accidents (1)
      off-peak    → more Road Works (9)

Run:
    python generate_timeseries.py
"""

import gc
import os
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from config import (
    WEEK_START, N_HOURS, BATCH_SIZE, FIELD_RES, RANDOM_SEED,
    INC_TYPES, TIMESERIES_DIR, STATIC_DIR,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TEMPORAL BASE PATTERN  — [7, 24] float32
#     Day 0 = Monday (2024-07-01),  values represent fractional congestion 0–1
# ══════════════════════════════════════════════════════════════════════════════

_WEEKDAY = np.array([
    0.10, 0.08, 0.07, 0.06, 0.08, 0.16,   # 00-05  night → pre-dawn
    0.32, 0.66, 0.91, 0.84, 0.67, 0.58,   # 06-11  morning peak  (8-9am ≈ 0.91)
    0.56, 0.54, 0.58, 0.65, 0.74, 0.93,   # 12-17  midday → PM buildup
    0.94, 0.86, 0.70, 0.54, 0.38, 0.22,   # 18-23  evening taper
], dtype=np.float32)

_SATURDAY = np.array([
    0.10, 0.08, 0.07, 0.06, 0.07, 0.10,
    0.18, 0.30, 0.44, 0.54, 0.62, 0.66,
    0.68, 0.66, 0.62, 0.60, 0.58, 0.60,
    0.62, 0.56, 0.48, 0.40, 0.32, 0.20,
], dtype=np.float32)

_SUNDAY = np.array([
    0.09, 0.07, 0.06, 0.06, 0.06, 0.08,
    0.12, 0.18, 0.26, 0.38, 0.46, 0.52,
    0.54, 0.52, 0.50, 0.47, 0.45, 0.47,
    0.50, 0.45, 0.38, 0.30, 0.22, 0.14,
], dtype=np.float32)

_PATTERNS = [_WEEKDAY, _WEEKDAY, _WEEKDAY, _WEEKDAY, _WEEKDAY, _SATURDAY, _SUNDAY]


def build_base_ts(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """[T] float32 base congestion intensity per hour."""
    return np.array([_PATTERNS[ts.dayofweek][ts.hour] for ts in timestamps],
                    dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  GLOBAL EVENT LAYERS  — all arrays [T=168]
# ══════════════════════════════════════════════════════════════════════════════

def build_events(timestamps: pd.DatetimeIndex) -> dict:
    """
    Constructs per-hour event arrays for the week 2024-07-01 → 2024-07-07.
    All values are physically motivated for Mumbai July (peak monsoon).
    """
    T = len(timestamps)

    # ── Hourly rainfall (mm/hr) ───────────────────────────────────────────────
    rain = np.zeros(T, dtype=np.float32)

    # Day 0 Mon — heavy afternoon shower
    rain[12:18] = [4., 18., 28., 22., 12., 5.]
    # Day 1 Tue — light drizzle
    rain[24+14:24+17] = [3., 8., 5.]
    # Day 2 Wed — very heavy; flood advisory (worst day)
    rain[48+12:48+20] = [5., 20., 45., 38., 25., 18., 10., 6.]
    # Day 3 Thu — moderate
    rain[72+14:72+18] = [6., 12., 9., 4.]
    # Day 4 Fri — heavy afternoon
    rain[96+13:96+19] = [8., 22., 32., 20., 12., 5.]
    # Day 5 Sat — morning drizzle + afternoon shower
    rain[120+8:120+11]  = [5., 8., 5.]
    rain[120+15:120+19] = [10., 18., 14., 8.]
    # Day 6 Sun — light
    rain[144+14:144+17] = [4., 7., 4.]

    # ── Monsoon flag (always 1 in July) ───────────────────────────────────────
    monsoon = np.ones(T, dtype=np.int8)

    # ── Local train disruption (1 = trains disrupted → people flood roads) ────
    # Wed 08:00-10:00 : signal failure on Western Railway
    # Fri 18:00-20:00 : track flooding
    train_dis = np.zeros(T, dtype=np.int8)
    train_dis[2*24 + 8  : 2*24 + 11] = 1
    train_dis[4*24 + 18 : 4*24 + 21] = 1

    # ── Holidays (none in this calendar week) ─────────────────────────────────
    holiday        = np.zeros(T, dtype=np.int8)
    school_holiday = np.zeros(T, dtype=np.int8)   # July = schools open in MH

    return {
        "hourly_rainfall_mm":     rain,
        "monsoon_active":         monsoon,
        "local_train_disruption": train_dis,
        "is_public_holiday":      holiday,
        "school_holiday":         school_holiday,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CONGESTION FIELD  [T, n_lat, n_lon]  float32
# ══════════════════════════════════════════════════════════════════════════════

def build_congestion_field(
    timestamps : pd.DatetimeIndex,
    events     : dict,
    lat_grid   : np.ndarray,    # [NL] sorted
    lon_grid   : np.ndarray,    # [NO] sorted
) -> np.ndarray:
    """
    Returns float32 array [T, NL, NO], values clipped to [0.03, 0.97].

    Steps:
      1. Urban decay factor derived from distance to Churchgate.
      2. AR(1) process in time, vectorised over the entire grid.
      3. Gaussian spatial smoothing (σ ≈ 1.3 cells ≈ 2.9 km).
      4. Rain and train-disruption add-ons applied cell-by-cell.
    """
    T  = len(timestamps)
    NL = len(lat_grid)
    NO = len(lon_grid)

    # ── urban decay factor [NL, NO] ───────────────────────────────────────────
    CENTRE_LAT, CENTRE_LON = 18.935, 72.827    # Churchgate
    lat_g, lon_g = np.meshgrid(lat_grid, lon_grid, indexing="ij")
    dlat_km = (lat_g - CENTRE_LAT) * 111.0
    dlon_km = (lon_g - CENTRE_LON) * 111.0 * np.cos(np.radians(19.2))
    dist_km = np.sqrt(dlat_km ** 2 + dlon_km ** 2)
    urban   = (0.28 + 0.72 * np.exp(-dist_km / 22.0)).clip(0.25, 1.0).astype(np.float32)
    # Result: South Mumbai ≈ 1.0, Vasai suburbs ≈ 0.30

    # ── base temporal signal [T] ───────────────────────────────────────────────
    base_ts = build_base_ts(timestamps)    # [T]

    # ── AR(1) — loop over T=168, vectorised over [NL, NO] grid ───────────────
    rng   = np.random.default_rng(RANDOM_SEED)
    noise = rng.standard_normal((T, NL, NO)).astype(np.float32) * 0.035

    alpha  = np.float32(0.68)
    field  = np.empty((T, NL, NO), dtype=np.float32)
    field[0] = (base_ts[0] * urban + noise[0]).clip(0.03, 0.97)

    for t in range(1, T):
        target   = base_ts[t] * urban                            # [NL, NO]
        field[t] = (alpha * field[t-1] + (1 - alpha) * target + noise[t]).clip(0.03, 0.97)

    del noise
    gc.collect()

    # ── Gaussian spatial smoothing ────────────────────────────────────────────
    # Smoothing makes neighbouring cells correlated (realistic traffic spread)
    for t in range(T):
        field[t] = gaussian_filter(field[t], sigma=1.3).clip(0.03, 0.97)

    # ── Rain add-on ───────────────────────────────────────────────────────────
    rain    = events["hourly_rainfall_mm"]           # [T]
    coastal = (lon_g < 72.86).astype(np.float32)     # coastal zones flood more [NL, NO]

    for t in range(T):
        r = float(rain[t])
        if r < 2.0:
            continue
        # Interpolated congestion increase based on rainfall intensity
        base_add  = float(np.interp(r, [2, 8, 18, 30], [0.04, 0.10, 0.20, 0.32]))
        flood_add = float(np.interp(r, [15, 25, 40],  [0.0,  0.10, 0.20])) if r >= 15 else 0.0
        field[t]  = (field[t] + base_add + flood_add * coastal).clip(0.03, 0.97)

    # ── Train disruption add-on ───────────────────────────────────────────────
    # When local trains are disrupted, road network absorbs extra demand (+10%)
    td = events["local_train_disruption"]            # [T]
    for t in range(T):
        if td[t]:
            field[t] = (field[t] + 0.10).clip(0.03, 0.97)

    return field       # [T, NL, NO]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  INCIDENT GENERATION  — fully vectorised
#     TomTom iconCategory : 1=Accident 4=Rain 6=Jam 8=RoadClosed 9=Works 11=Flood
#     TomTom magnitudeOfDelay : 0=none 1=minor 2=moderate 3=major 4=huge
# ══════════════════════════════════════════════════════════════════════════════

_INC_ARR = np.array(INC_TYPES, dtype=np.int8)    # [1, 4, 6, 8, 9, 11]

def generate_incidents(
    edge_cong : np.ndarray,   # [T, N]  float32
    rain      : np.ndarray,   # [T]     float32
    train_dis : np.ndarray,   # [T]     int8
    base_ts   : np.ndarray,   # [T]     float32  (used as peak proxy)
    rng       : np.random.Generator,
) -> tuple:
    """
    Returns (incident, incident_type, incident_severity) each [T, N] int8.

    Incident probability is causally derived:
      p = 0.015  +  0.07 × congestion  +  0.05 × (rain/30)  +  0.04 × train_dis
    Types shift depending on weather / time of day.
    Severity (magnitudeOfDelay) is proportional to congestion.
    """
    T, N = edge_cong.shape

    # ── sample incident flag ──────────────────────────────────────────────────
    r2d = rain[:, np.newaxis].astype(np.float32)     # [T, 1]
    td2d = train_dis.astype(np.float32)[:, np.newaxis]

    p_inc = (
        0.015
        + 0.07  * edge_cong
        + 0.05  * np.clip(r2d / 30.0, 0, 1)
        + 0.04  * td2d
    ).clip(0, 0.28).astype(np.float32)

    incident = (rng.random((T, N), dtype=np.float32) < p_inc).astype(np.int8)

    # ── classify hour conditions (per timestep) ───────────────────────────────
    heavy_rain = rain > 15                      # [T] bool
    mild_rain  = (rain > 2) & ~heavy_rain       # [T] bool
    is_peak    = base_ts > 0.65                 # [T] bool  (rush hour proxy)

    # ── broadcast to [T, N] condition masks (only where incident == 1) ────────
    inc_1 = incident == 1
    m_heavy = heavy_rain[:, None] &  inc_1
    m_mild  = mild_rain[:, None]  &  inc_1  & ~m_heavy
    m_peak  = is_peak[:, None]    &  inc_1  & ~m_heavy & ~m_mild
    m_norm  =                        inc_1  & ~m_heavy & ~m_mild & ~m_peak

    inc_type = np.zeros((T, N), dtype=np.int8)

    # Probability weights over _INC_ARR = [1, 4, 6, 8, 9, 11]
    #                                     Acc Rain Jam Cls Wrk Fld
    conditions = [
        (m_heavy, [0.10, 0.18, 0.28, 0.04, 0.04, 0.36]),  # heavy rain → flood/jam
        (m_mild,  [0.20, 0.26, 0.34, 0.03, 0.10, 0.07]),  # mild rain  → rain/jam
        (m_peak,  [0.30, 0.04, 0.50, 0.04, 0.08, 0.04]),  # peak hour  → jam/accident
        (m_norm,  [0.35, 0.04, 0.35, 0.04, 0.18, 0.04]),  # off-peak   → accident/works
    ]

    for mask, probs in conditions:
        n_cells = int(mask.sum())
        if n_cells == 0:
            continue
        chosen = rng.choice(_INC_ARR, size=n_cells, p=probs)
        inc_type[mask] = chosen

    # ── severity: proportional to congestion level ────────────────────────────
    inc_sev = np.zeros((T, N), dtype=np.int8)
    inc_sev[inc_1 & (edge_cong <= 0.40)] = 1   # minor
    inc_sev[inc_1 & (edge_cong >  0.40) & (edge_cong <= 0.65)] = 2   # moderate
    inc_sev[inc_1 & (edge_cong >  0.65) & (edge_cong <= 0.82)] = 3   # major
    inc_sev[inc_1 & (edge_cong >  0.82)] = 4   # huge

    # small stochastic bump ±1 (20% of incidents)
    bump_mask = inc_1 & (rng.random((T, N), dtype=np.float32) < 0.20)
    direction = np.where(
        rng.random((T, N), dtype=np.float32) < 0.5, 1, -1
    ).astype(np.int8)
    inc_sev = np.clip(
        inc_sev + (direction * bump_mask.astype(np.int8)),
        0, 4
    ).astype(np.int8)

    return incident, inc_type, inc_sev


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PROCESS ONE BATCH OF EDGES  →  long-format DataFrame [N × T rows]
# ══════════════════════════════════════════════════════════════════════════════

def process_batch(
    batch      : pd.DataFrame,
    field      : np.ndarray,        # [T, NL, NO]
    lat_grid   : np.ndarray,        # [NL]
    lon_grid   : np.ndarray,        # [NO]
    timestamps : pd.DatetimeIndex,
    events     : dict,
    base_ts    : np.ndarray,
    zone_exp_inc : np.ndarray,      # [T, n_zones_global]  — precomputed expected counts
    rng        : np.random.Generator,
) -> pd.DataFrame:
    """
    Produces N × T rows (all 168 timesteps for each edge, edge-major order).
    """
    T  = len(timestamps)
    N  = len(batch)

    # ── look up congestion at edge centroid from field (nearest cell) ─────────
    lat_idx = np.searchsorted(lat_grid, batch["lat"].values).clip(0, len(lat_grid) - 1)
    lon_idx = np.searchsorted(lon_grid, batch["lon"].values).clip(0, len(lon_grid) - 1)
    edge_cong = field[:, lat_idx, lon_idx].copy()      # [T, N]

    # ── apply road-type susceptibility ───────────────────────────────────────
    susc = batch["susceptibility"].values.astype(np.float32)   # [N]
    edge_cong = (edge_cong * susc[np.newaxis, :]).clip(0.03, 0.97)

    # ── TomTom-like speed / travel time ──────────────────────────────────────
    ffs     = batch["free_flow_speed"].values.astype(np.float32)    # [N] km/h
    length  = batch["road_length"].values.astype(np.float32)        # [N] metres
    ff_tt   = (length / (ffs / 3.6)).clip(min=1.0)                  # [N] seconds

    # speed_ratio: add small per-edge Gaussian noise (±2%) for realism
    speed_ratio = (
        (1.0 - edge_cong) +
        rng.standard_normal((T, N)).astype(np.float32) * 0.02
    ).clip(0.05, 1.0)                                               # [T, N]

    current_speed = (ffs[np.newaxis, :] * speed_ratio).astype(np.float32)   # [T, N] km/h
    current_tt    = (ff_tt[np.newaxis, :] / speed_ratio).astype(np.float32) # [T, N] sec
    tt_ratio      = (current_tt / ff_tt[np.newaxis, :]).clip(1.0, 10.0)     # [T, N]
    cong_level    = (1.0 - speed_ratio).clip(0.0, 0.97)                     # [T, N]
    delay_sec     = (current_tt - ff_tt[np.newaxis, :]).clip(0.0)           # [T, N]

    # confidence: TomTom confidence is higher when traffic is free-flowing
    confidence = (
        0.68 + 0.28 * speed_ratio
        + rng.standard_normal((T, N)).astype(np.float32) * 0.018
    ).clip(0.38, 0.99)

    # road_closure: physical congestion threshold OR rare random event
    road_closure = (
        (edge_cong > 0.95) |
        (rng.random((T, N), dtype=np.float32) < 0.0004)
    ).astype(np.int8)

    # When road is closed: zero movement
    mask_closed = road_closure == 1
    current_speed[mask_closed] = ffs[np.newaxis, :].repeat(T, 0)[mask_closed] * 0.04
    current_tt[mask_closed]    = ff_tt[np.newaxis, :].repeat(T, 0)[mask_closed] * 14.0
    tt_ratio[mask_closed]      = 10.0
    cong_level[mask_closed]    = 0.97

    # ── incidents ─────────────────────────────────────────────────────────────
    rain_arr = events["hourly_rainfall_mm"]
    td_arr   = events["local_train_disruption"]
    incident, inc_type, inc_sev = generate_incidents(
        edge_cong, rain_arr, td_arr, base_ts, rng
    )

    # incidents_nearby: expected incident count in this zone at this time
    # (precomputed zone-level expectation, consistent across all batches)
    zone_ids        = batch["zone_id"].values.astype(int)    # [N]
    incidents_nearby = zone_exp_inc[:, zone_ids].astype(np.int16)  # [T, N]

    # ── cyclic time encodings ─────────────────────────────────────────────────
    hours   = np.array([ts.hour      for ts in timestamps], dtype=np.float32)
    days    = np.array([ts.dayofweek for ts in timestamps], dtype=np.float32)
    tod_sin = np.sin(2 * np.pi * hours / 24)
    tod_cos = np.cos(2 * np.pi * hours / 24)
    dow_sin = np.sin(2 * np.pi * days  /  7)
    dow_cos = np.cos(2 * np.pi * days  /  7)

    # ── build long DataFrame  [N*T rows, edge-major] ──────────────────────────
    # flat(arr_TN) : [T,N] → .T gives [N,T] → .ravel() gives [N*T] edge-major
    # tile_T(x)    : [T] → repeated N times  → [N*T]
    # rep_N(x)     : [N] → each element T times → [N*T]

    def flat(a):        return a.T.ravel()
    def tile_T(a):      return np.tile(a, N)
    def rep_N(a):       return np.repeat(a, T)

    df = pd.DataFrame({
        # ── identifiers ───────────────────────────────────────────────────────
        "edge_id"   : rep_N(batch["edge_id"].values),
        "timestamp" : np.tile(timestamps, N),

        # ── TomTom Flow ───────────────────────────────────────────────────────
        "current_speed"          : flat(current_speed).round(2),
        "free_flow_speed"        : rep_N(ffs),
        "current_travel_time"    : flat(current_tt).round(1),
        "free_flow_travel_time"  : rep_N(ff_tt.round(1)),
        "confidence"             : flat(confidence).round(3),
        "road_closure"           : flat(road_closure),

        # ── TomTom Incidents ──────────────────────────────────────────────────
        "incident"               : flat(incident),
        "incident_type"          : flat(inc_type),       # TomTom iconCategory
        "incident_severity"      : flat(inc_sev),        # TomTom magnitudeOfDelay
        "incidents_nearby"       : flat(incidents_nearby),

        # ── Context / external events ─────────────────────────────────────────
        "hourly_rainfall_mm"     : tile_T(events["hourly_rainfall_mm"]),
        "monsoon_active"         : tile_T(events["monsoon_active"]),
        "local_train_disruption" : tile_T(events["local_train_disruption"]),
        "is_public_holiday"      : tile_T(events["is_public_holiday"]),
        "school_holiday"         : tile_T(events["school_holiday"]),

        # ── Derived features ──────────────────────────────────────────────────
        "travel_time_ratio"      : flat(tt_ratio).round(4),
        "congestion_level"       : flat(cong_level).round(4),
        "delay_seconds"          : flat(delay_sec).round(1),
        "speed_ratio"            : flat(speed_ratio).round(4),

        # ── Cyclic time encodings ─────────────────────────────────────────────
        "time_of_day_sin"        : tile_T(tod_sin.round(5)),
        "time_of_day_cos"        : tile_T(tod_cos.round(5)),
        "day_of_week_sin"        : tile_T(dow_sin.round(5)),
        "day_of_week_cos"        : tile_T(dow_cos.round(5)),
    })

    # ── cast flag columns to int8 to save space ───────────────────────────────
    for col in ["road_closure", "incident", "incident_type", "incident_severity",
                "monsoon_active", "local_train_disruption",
                "is_public_holiday", "school_holiday"]:
        df[col] = df[col].astype(np.int8)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(TIMESERIES_DIR, exist_ok=True)

    # ── load static features ──────────────────────────────────────────────────
    static_path = f"{STATIC_DIR}/edges_static.parquet"
    print(f"Loading static features from {static_path} …")
    df_static = pd.read_parquet(static_path)
    n_edges   = len(df_static)
    n_zones   = int(df_static["zone_id"].max()) + 1
    print(f"  {n_edges:,} edges,  {n_zones} zones")

    # ── timestamps ────────────────────────────────────────────────────────────
    timestamps = pd.date_range(start=WEEK_START, periods=N_HOURS, freq="h")
    print(f"  Timestamps : {timestamps[0]}  →  {timestamps[-1]}")

    # ── global event arrays ───────────────────────────────────────────────────
    events  = build_events(timestamps)
    base_ts = build_base_ts(timestamps)

    # ── build congestion field ────────────────────────────────────────────────
    lat_min = float(df_static["lat"].min()) - FIELD_RES
    lat_max = float(df_static["lat"].max()) + FIELD_RES
    lon_min = float(df_static["lon"].min()) - FIELD_RES
    lon_max = float(df_static["lon"].max()) + FIELD_RES

    lat_grid = np.arange(lat_min, lat_max + FIELD_RES / 2, FIELD_RES, dtype=np.float32)
    lon_grid = np.arange(lon_min, lon_max + FIELD_RES / 2, FIELD_RES, dtype=np.float32)

    print(f"\nBuilding congestion field [{N_HOURS}, {len(lat_grid)}, {len(lon_grid)}] …")
    t0    = time.perf_counter()
    field = build_congestion_field(timestamps, events, lat_grid, lon_grid)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # ── precompute zone-level expected incident count [T, n_zones] ────────────
    # This ensures incidents_nearby is consistent across all batches
    print("Precomputing zone incident expectations …")
    zone_centres = (
        df_static.groupby("zone_id")[["lat", "lon"]]
        .mean()
        .sort_index()
        .reset_index()
    )
    zone_lat_idx = np.searchsorted(lat_grid, zone_centres["lat"].values).clip(0, len(lat_grid) - 1)
    zone_lon_idx = np.searchsorted(lon_grid, zone_centres["lon"].values).clip(0, len(lon_grid) - 1)
    zone_cong    = field[:, zone_lat_idx, zone_lon_idx]   # [T, n_zones]

    # edges per zone
    zone_sizes = df_static["zone_id"].value_counts().sort_index().values  # [n_zones]

    # expected incident count = p_inc * n_edges_in_zone
    rain2d  = events["hourly_rainfall_mm"][:, np.newaxis]
    td2d    = events["local_train_disruption"].astype(np.float32)[:, np.newaxis]
    p_zone  = (0.015 + 0.07 * zone_cong + 0.05 * np.clip(rain2d / 30, 0, 1) + 0.04 * td2d).clip(0, 0.28)
    zone_exp_inc = np.round(p_zone * zone_sizes[np.newaxis, :]).astype(np.int16)  # [T, n_zones]

    # ── batch processing ──────────────────────────────────────────────────────
    n_batches = (n_edges + BATCH_SIZE - 1) // BATCH_SIZE
    rng       = np.random.default_rng(RANDOM_SEED + 1)

    print(f"\nGenerating time-series: {n_batches} batches × up to {BATCH_SIZE} edges × {N_HOURS} h")
    print(f"Output directory: {TIMESERIES_DIR}/\n")

    total_rows = 0
    t_start    = time.perf_counter()

    for b in range(n_batches):
        lo    = b * BATCH_SIZE
        hi    = min(lo + BATCH_SIZE, n_edges)
        batch = df_static.iloc[lo:hi].reset_index(drop=True)

        df_batch = process_batch(
            batch, field, lat_grid, lon_grid,
            timestamps, events, base_ts, zone_exp_inc, rng,
        )

        out_path = f"{TIMESERIES_DIR}/batch_{b:04d}.parquet"
        df_batch.to_parquet(out_path, index=False, compression="snappy")
        total_rows += len(df_batch)

        elapsed = time.perf_counter() - t_start
        eta     = elapsed / (b + 1) * (n_batches - b - 1)
        print(
            f"  batch {b+1:3d}/{n_batches}  edges {lo:,}–{hi:,}  "
            f"rows={len(df_batch):,}  "
            f"elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s"
        )

        del df_batch
        gc.collect()

    print(f"\n✓  Done.")
    print(f"   Total rows       : {total_rows:,}")
    print(f"   Parquet batches  : {n_batches}  ({TIMESERIES_DIR}/batch_XXXX.parquet)")
    print(f"   Static features  : {STATIC_DIR}/edges_static.parquet\n")

    _print_sample_stats(df_static)


def _print_sample_stats(df_static: pd.DataFrame) -> None:
    """Load batch_0000 and print a dataset snapshot for verification."""
    import glob
    batches = sorted(glob.glob(f"{TIMESERIES_DIR}/batch_*.parquet"))
    if not batches:
        return

    df  = pd.read_parquet(batches[0])
    sta = pd.read_parquet(f"{STATIC_DIR}/edges_static.parquet")

    print("─" * 70)
    print("SAMPLE: edges_static.parquet  .head(3)")
    print("─" * 70)
    print(sta.head(3).to_string(index=False))
    print(f"\nColumns : {list(sta.columns)}")
    print(f"Shape   : {sta.shape}")

    print("\n" + "─" * 70)
    print(f"SAMPLE: {batches[0]}  .head(5)")
    print("─" * 70)
    print(df.head(5).to_string(index=False))
    print(f"\nColumns : {list(df.columns)}")
    print(f"Shape   : {df.shape}  (one batch = {len(df)//168} edges × 168 h)")

    print("\n" + "─" * 70)
    print("TIMESERIES STATS (batch 0)")
    print("─" * 70)
    cols = ["travel_time_ratio", "congestion_level", "current_speed",
            "incident", "incidents_nearby", "hourly_rainfall_mm"]
    print(df[cols].describe().round(3).to_string())


if __name__ == "__main__":
    main()
