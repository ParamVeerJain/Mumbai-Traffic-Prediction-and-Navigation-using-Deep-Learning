# Mumbai Traffic Dataset — Spatial-Temporal Generator

---

## Dependencies

Install everything with one command:

```bash
pip install osmnx>=1.9,<2.0 pandas>=2.1 numpy>=1.24 geopandas>=0.14 scipy>=1.11 pyarrow>=14.0 shapely>=2.0 networkx>=3.0
```

Or via the requirements file:

```bash
pip install -r requirements.txt
```

Tested on Python 3.10+. OSMnx will also pull in `requests`, `fiona`, `pyproj` automatically.

---

## How to Run (in order)

### Step 1 — Fetch the OSM road graph

```bash
python fetch_graph.py
```

- Downloads the Mumbai road network from OpenStreetMap via the Overpass API
- Makes 54 API calls (9 lat × 6 lon points, each with a 6 km radius circle)
- Composes all sub-graphs into one unified graph with `ox.compose_all()`
- Saves to `data/graph/mumbai.graphml`
- First run: **10–25 minutes** depending on your internet / Overpass load
- Repeat runs: **~5 seconds** (OSMnx caches API responses on disk automatically)

### Step 2 — Build static edge features

```bash
python build_static.py
```

- Loads `mumbai.graphml` and extracts one row per road edge
- Parses and imputes OSM tags (`highway`, `lanes`, `maxspeed`, `oneway`, `length`)
- Computes derived static features: `signals_per_km`, `zone_id`, `corridor_id`, etc.
- Saves to `data/static/edges_static.parquet`
- Runs in **~30–90 seconds** (pure in-memory pandas/networkx, no API calls)

### Step 3 — Generate the time-series dataset

```bash
python generate_timeseries.py
```

- Builds a spatially coherent congestion field over the full network for 168 hours
- Processes edges in batches of 5,000 to stay within RAM limits
- Writes one parquet file per batch to `data/timeseries/batch_XXXX.parquet`
- Runtime: **5–20 minutes** depending on the number of edges (~50k–140k)

---

## Directory Structure

```
traffic_dataset/
│
├── requirements.txt
├── config.py                  ← all constants: grid, time params, road defaults
├── fetch_graph.py             ← Step 1: OSM download
├── build_static.py            ← Step 2: static feature extraction
├── generate_timeseries.py     ← Step 3: synthetic spatial-temporal data
│
└── data/
    ├── graph/
    │   └── mumbai.graphml           ← full composed OSM graph
    │
    ├── static/
    │   └── edges_static.parquet     ← 1 row per edge, all static features
    │
    └── timeseries/
        ├── batch_0000.parquet       ← edges 0–4,999    × 168 hours
        ├── batch_0001.parquet       ← edges 5,000–9,999 × 168 hours
        ├── batch_0002.parquet
        └── ...                      ← as many as needed for all edges
```

---

## Output Files

### `data/static/edges_static.parquet`

One row per road edge. Shape: `[N_edges, 17]`

```
edge_id               → unique string ID: "u_v_key"  e.g. "123456_123457_0"
u, v                  → OSM node IDs (start / end of edge)
lat, lon              → centroid of the edge geometry (float32)
road_type             → OSM highway tag: motorway / primary / residential / etc.
road_type_enc         → ordinal int8: motorway=8 ... residential=2 ... service=1
num_lanes             → int8, imputed from road_type if OSM tag missing
oneway                → bool
free_flow_speed       → km/h float32 (from OSM maxspeed or road_type default)
road_length           → metres float32 (OSM geometry length)
traffic_signal_count  → int8: 0/1/2 signals at u or v node
intersection_count    → int8: how many real junctions this edge touches
signals_per_km        → float32: traffic_signal_count / (road_length / 1000)
susceptibility        → float32: how much zone congestion affects this road type
zone_id               → int32: 0.02° grid cell index (~2.2 km cells)
corridor_id           → int16: 0.10° super-zone index (~11 km corridors)
```

Sample `.head(3)`:

```
edge_id              u        v  lat     lon     road_type  road_type_enc  num_lanes  oneway  free_flow_speed  road_length  traffic_signal_count  intersection_count  signals_per_km  susceptibility  zone_id  corridor_id
123456_123457_0  123456  123457  19.03  72.84  primary           6          2     False     60.0        342.5                1                   2           2.92        1.05        11           2
234561_234562_0  234561  234562  19.07  72.89  residential       2          1      True     30.0         95.3                0                   0           0.00        0.70        14           2
345671_345672_0  345671  345672  19.22  72.82  motorway          8          6      True    100.0       1204.0                0                   0           0.00        0.78         8           1
```

---

### `data/timeseries/batch_XXXX.parquet`

One row per (edge × hour). Each batch has `min(BATCH_SIZE, remaining_edges) × 168` rows.

All 140k edges × 168 hours = **~23.5 million rows** across all batch files.

```
edge_id                  → joins to edges_static on this key

── TomTom Flow (synthetic, 1:1 with real TomTom API fields) ──────────────────
current_speed            → float32  km/h actual speed at this hour
free_flow_speed          → float32  km/h free-flow baseline (constant per edge)
current_travel_time      → float32  seconds to traverse this edge now
free_flow_travel_time    → float32  seconds with no traffic (constant per edge)
confidence               → float32  0.38–0.99 (higher on clear roads)
road_closure             → int8     0 or 1 (TomTom boolean)

── TomTom Incidents (synthetic, matching TomTom iconCategory / magnitude) ────
incident                 → int8     0=none  1=has incident
incident_type            → int8     TomTom iconCategory:
                                    0=none 1=Accident 4=Rain 6=Jam
                                    8=RoadClosed 9=Works 11=Flooding
incident_severity        → int8     TomTom magnitudeOfDelay:
                                    0=none 1=minor 2=moderate 3=major 4=huge
incidents_nearby         → int16    expected count of incidents in same zone this hour

── External / Mumbai-specific events ─────────────────────────────────────────
hourly_rainfall_mm       → float32  0–45 mm/hr (July monsoon, realistic daily profile)
monsoon_active           → int8     always 1 (July week)
local_train_disruption   → int8     0/1 (Wed 08-10, Fri 18-20 — train signal failure)
is_public_holiday        → int8     0 (none in this calendar week)
school_holiday           → int8     0 (schools open in Maharashtra in July)

── Derived / model-ready features ────────────────────────────────────────────
travel_time_ratio        → float32  current_tt / free_flow_tt  [1.0 – 10.0]  PRIMARY TARGET
congestion_level         → float32  1 - speed_ratio            [0.0 – 0.97]
delay_seconds            → float32  current_tt - free_flow_tt  ≥ 0
speed_ratio              → float32  current_speed / free_flow_speed [0.05–1.0]
time_of_day_sin/cos      → float32  cyclic hour encoding
day_of_week_sin/cos      → float32  cyclic day encoding
```

Sample `.head(5)` (one edge, five consecutive hours):

```
edge_id              timestamp             current_speed  free_flow_speed  current_travel_time  free_flow_travel_time  confidence  road_closure  incident  incident_type  incident_severity  incidents_nearby  hourly_rainfall_mm  monsoon_active  local_train_disruption  travel_time_ratio  congestion_level  delay_seconds  speed_ratio  time_of_day_sin  time_of_day_cos
123456_123457_0  2024-07-01 00:00:00       55.8            60.0             22.1                  20.6                  0.921         0             0              0                  0                  2                0.0              1                 0                1.073             0.070           1.5          0.930           0.000            1.000
123456_123457_0  2024-07-01 01:00:00       56.4            60.0             21.8                  20.6                  0.933         0             0              0                  0                  2                0.0              1                 0                1.058             0.060           1.2          0.940           0.259            0.966
123456_123457_0  2024-07-01 08:00:00       22.1            60.0             56.2                  20.6                  0.740         0             1              6                  3                  8                0.0              1                 0                2.728             0.632          35.6          0.368           1.000            0.000
123456_123457_0  2024-07-01 14:00:00       38.4            60.0             32.4                  20.6                  0.832         0             0              0                  0                  4               22.0              1                 0                1.573             0.360          11.8          0.640          -0.866            0.500
123456_123457_0  2024-07-01 20:00:00       11.8            60.0             105.7                 20.6                  0.649         0             1              6                  4                 14                0.0              1                 1                5.132             0.803          85.1          0.197          -0.866           -0.500
```

---

## What Kind of Data You Actually Get

The dataset is **spatially coherent** — roads close to each other have correlated congestion, not independent random values. Roads in South Mumbai (near Churchgate) are structurally busier than Vasai suburbs.

The data is **temporally coherent** — each edge has an AR(1) time series with realistic weekday/weekend patterns. Rush hours (8–10am, 5–8pm) produce high `travel_time_ratio`. Off-peak (1–4am) produces ratio near 1.0.

The events are **physically causal**:
- Heavy rain → higher `incident_type=4` (Rain) and `incident_type=11` (Flooding), slower speeds
- Train disruption Wed 08-10 / Fri 18-20 → road congestion spikes across all edges
- Road type governs magnitude: motorways absorb congestion better than primary/secondary roads

---

## How to Load for Training

```python
import pandas as pd
import glob

# Load all time-series batches (lazy / streaming friendly)
batches = sorted(glob.glob("data/timeseries/batch_*.parquet"))
df_ts   = pd.concat([pd.read_parquet(b) for b in batches])

# Join static features
df_sta  = pd.read_parquet("data/static/edges_static.parquet")
df      = df_ts.merge(df_sta, on="edge_id", how="left")

# Sort for sliding window construction
df      = df.sort_values(["edge_id", "timestamp"]).reset_index(drop=True)
```

Or load one batch at a time to stay within RAM:

```python
for batch_path in sorted(glob.glob("data/timeseries/batch_*.parquet")):
    df = pd.read_parquet(batch_path)
    df = df.merge(df_sta, on="edge_id", how="left")
    # ... build windows, push to DataLoader, etc.
```

---

## RAM & Disk Estimates

| File | Rows | Approx size on disk |
|---|---|---|
| `edges_static.parquet` | ~140k | ~15 MB |
| single `batch_XXXX.parquet` | ~840k (5k edges × 168) | ~35–50 MB |
| all batches combined | ~23.5M | ~1.2–1.8 GB |
| full joined DataFrame in RAM | ~23.5M | ~4–6 GB |

Always process one batch at a time during training. Never load all batches into RAM simultaneously.
