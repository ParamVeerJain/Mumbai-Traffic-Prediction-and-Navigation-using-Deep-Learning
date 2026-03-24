# config.py
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "Models")

# ── LSTM ──────────────────────────────────────────────────────
LSTM_PATH    = os.path.join(MODELS_DIR, "lstm.pth")
SEQ_LEN      = 22
PRED_LEN     = 9
INPUT_DIM    = 10
HIDDEN       = 128
LAYERS       = 2
EMB_DIM      = 16
DROPOUT      = 0.2

FEATURE_COLS = [
    "hour_sine", "hour_cos",
    "day_sin",   "day_cos",
    "travel_time_log_Scaled",
    "rain",      "incident",
    "congestion",
    "length_log_Scaled",
    "speed_ratio",
]

# ── Routing ───────────────────────────────────────────────────
DEFAULT_SPEED_KMH = 40.0
MAX_HORIZON       = 9

# ── OSM download ──────────────────────────────────────────────
OSM_NETWORK_TYPE  = "drive"
OSM_DIST_M        = 3000

# ── PPO ───────────────────────────────────────────────────────
PPO_GAMMA        = 0.99
PPO_LR           = 3e-4
PPO_CLIP_EPS     = 0.2
PPO_EPOCHS       = 4
PPO_HIDDEN       = 128
PPO_CHECKPOINT   = os.path.join(MODELS_DIR, "ppo_policy.pth")