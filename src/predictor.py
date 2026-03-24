# src/predictor.py
"""
LSTM inference — given a graph + departure datetime,
updates every edge's predicted_speed and travel_time_s.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from config import (LSTM_PATH, SEQ_LEN, PRED_LEN, INPUT_DIM,
                    HIDDEN, LAYERS, EMB_DIM, DROPOUT, FEATURE_COLS)
from src.features import build_edge_sequence


# ── Model definition (must match training) ───────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden=HIDDEN, layers=LAYERS,
                 pred_len=PRED_LEN, dropout=DROPOUT,
                 num_roads=1000, emb_dim=EMB_DIM):
        super().__init__()
        self.road_emb = nn.Embedding(num_roads + 1, emb_dim, padding_idx=0)
        self.lstm     = nn.LSTM(input_size  = input_dim + emb_dim,
                                hidden_size = hidden,
                                num_layers  = layers,
                                dropout     = dropout,
                                batch_first = True)
        self.norm     = nn.LayerNorm(hidden)
        self.shared   = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32),    nn.GELU(),
        )
        self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(pred_len)])

    def forward(self, x, road_ids):
        emb = self.road_emb(road_ids).unsqueeze(1).expand(-1, x.size(1), -1)
        out, _ = self.lstm(torch.cat([x, emb], dim=-1))
        out     = self.norm(out[:, -1, :])
        s       = self.shared(out)
        return torch.cat([h(s) for h in self.heads], dim=1)


# ── Predictor class ───────────────────────────────────────────────────────────
class TrafficPredictor:
    def __init__(self, road_to_idx: dict, num_roads: int,
                 device=None, df_history=None):
        self.road_to_idx = road_to_idx
        self.num_roads   = num_roads
        self.df_history  = df_history
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model       = self._load_model(num_roads)

    def _load_model(self, num_roads):
        model = LSTMModel(num_roads=num_roads)
        if os.path.exists(LSTM_PATH):
            state = torch.load(LSTM_PATH, map_location=self.device)
            model.load_state_dict(state)
            print(f"[predictor] Loaded weights from {LSTM_PATH}")
        else:
            print("[predictor] lstm.pth not found — using untrained model (dummy mode)")
        model.to(self.device).eval()
        return model

    def predict_graph(self, G, departure_dt: pd.Timestamp, horizon_step: int):
        """
        For every edge in G, predict the speed_ratio at horizon_step (1–9),
        then update G[u][v]['predicted_speed'] and G[u][v]['travel_time_s'].

        horizon_step = 1 means the NEXT hour, 2 means two hours ahead, etc.
        Clamped to [1, PRED_LEN].
        """
        h_idx = int(np.clip(horizon_step, 1, PRED_LEN)) - 1   # 0-indexed

        edges   = list(G.edges(data=True))
        seqs    = []
        road_ids = []

        for u, v, attr in edges:
            edge_data = dict(attr)
            edge_data["u"] = u
            edge_data["v"] = v
            seq = build_edge_sequence(edge_data, departure_dt, self.df_history)
            seqs.append(seq)

            key     = (float(G.nodes[u]["lat"]), float(G.nodes[u]["lon"]))
            road_id = self.road_to_idx.get((u, v), 0)
            road_id = int(np.clip(road_id, 0, self.num_roads))
            road_ids.append(road_id)

        X  = torch.tensor(np.array(seqs), dtype=torch.float32).to(self.device)
        R  = torch.tensor(road_ids,       dtype=torch.long   ).to(self.device)

        CHUNK = 512
        preds_all = []
        with torch.no_grad():
            for i in range(0, len(X), CHUNK):
                preds_all.append(
                    self.model(X[i:i+CHUNK], R[i:i+CHUNK]).cpu().numpy()
                )

        preds = np.vstack(preds_all)   # (E, PRED_LEN)

        for idx, (u, v, attr) in enumerate(edges):
            sr    = float(np.clip(preds[idx, h_idx], 0.05, 1.0))
            ff    = attr.get("free_flow_speed", 40.0)
            speed = sr * ff
            length = attr.get("length_m", 100.0)
            tt    = length / (speed * 1000/3600 + 1e-6)
            G[u][v]["predicted_speed"] = speed
            G[u][v]["speed_ratio"]     = sr
            G[u][v]["travel_time_s"]   = tt

        print(f"[predictor] Updated {len(edges)} edges for horizon t+{horizon_step}")
        return G