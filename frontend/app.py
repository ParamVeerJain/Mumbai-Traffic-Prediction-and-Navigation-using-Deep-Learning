import base64
import json
import os
from functools import lru_cache
from typing import Dict, List

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# DATA_ROOT = os.path.join(PROJECT_ROOT, "backend", "data generation", "data")
DATA_ROOT = r'E:\Projects\Major 2.0\Mumbai-Traffic-Prediction-and-Navigation-using-Deep-Learning\backend\data generation\data'
PNG_ROOT = os.path.join(DATA_ROOT, "pngs")
PRED_ROOT = os.path.join(DATA_ROOT, "predictions")


def _logo() -> str:
    svg = """
    <svg xmlns='http://www.w3.org/2000/svg' width='260' height='48'>
      <rect width='260' height='48' rx='10' ry='10' fill='#0f172a'/>
      <text x='14' y='31' fill='#38bdf8' font-size='20' font-family='Arial'>Mumbai Traffic Pulse</text>
    </svg>
    """.strip()
    payload = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{payload}"


@lru_cache(maxsize=64)
def _img_data(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _horizons_available() -> List[int]:
    out = []
    if not os.path.isdir(PRED_ROOT):
        return out
    for name in os.listdir(PRED_ROOT):
        if name.startswith("horizon_tplus_") and name.endswith(".parquet"):
            out.append(int(name.replace("horizon_tplus_", "").replace(".parquet", "")))
    return sorted(out)


def _frames_json_path(region: str, horizon: int) -> str:
    return os.path.join(PNG_ROOT, f"tplus_{horizon:02d}", region, "frames.json")


def _safe_load_frames(region: str, horizon: int) -> List[Dict]:
    path = _frames_json_path(region, horizon)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Mumbai Traffic Pulse"

horizons = _horizons_available()
horizon_opts = [{"label": f"t+{h}", "value": h} for h in horizons] or [{"label": "No data", "value": 1}]

app.layout = dbc.Container(
    fluid=True,
    style={"padding": "16px"},
    children=[
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "14px"},
            children=[
                html.Img(src=_logo(), style={"height": "48px"}),
                html.H2("Actual vs Predicted Congestion Simulator", style={"margin": 0}),
            ],
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Model"),
                        dcc.Dropdown(
                            id="model",
                            options=[{"label": "Temporal (default)", "value": "temporal"}],
                            value="temporal",
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Region"),
                        dcc.Dropdown(
                            id="region",
                            options=[{"label": "Mumbai", "value": "mumbai"}, {"label": "Goregaon", "value": "goregaon"}],
                            value="mumbai",
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Horizon"),
                        dcc.Dropdown(id="horizon", options=horizon_opts, value=horizon_opts[0]["value"], clearable=False),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Speed (ms/frame)"),
                        dcc.Slider(id="speed", min=100, max=2000, step=100, value=600),
                    ],
                    md=3,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Button("Pause/Resume", id="toggle-play", color="warning"), md=2),
                dbc.Col(html.Div(id="time-label", style={"fontSize": "20px", "fontWeight": 700}), md=10),
            ],
            style={"marginTop": "10px"},
        ),
        dcc.Interval(id="ticker", interval=600, n_intervals=0),
        dcc.Store(id="play-state", data={"paused": False}),
        dcc.Store(id="frame-index", data={"i": 0}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Actual"),
                        html.Img(id="actual-map", style={"width": "100%", "height": "80vh", "objectFit": "contain"}),
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        html.H4("Prediction"),
                        html.Img(id="pred-map", style={"width": "100%", "height": "80vh", "objectFit": "contain"}),
                    ],
                    md=6,
                ),
            ],
            style={"marginTop": "14px"},
        ),
        html.Div(id="error-box", style={"color": "#ff6b6b", "marginTop": "8px"}),
    ],
)


@app.callback(Output("play-state", "data"), Input("toggle-play", "n_clicks"), State("play-state", "data"), prevent_initial_call=True)
def toggle_play(_, st):
    return {"paused": not bool(st["paused"])}


@app.callback(Output("ticker", "interval"), Input("speed", "value"))
def set_speed(speed):
    return int(speed)


@app.callback(
    Output("actual-map", "src"),
    Output("pred-map", "src"),
    Output("time-label", "children"),
    Output("frame-index", "data"),
    Output("error-box", "children"),
    Input("ticker", "n_intervals"),
    Input("region", "value"),
    Input("horizon", "value"),
    Input("model", "value"),
    State("play-state", "data"),
    State("frame-index", "data"),
)
def update_frame(_, region, horizon, model, play_state, frame_state):
    if model != "temporal":
        return "", "", "", frame_state, "Selected model is unavailable. Please select Temporal."

    frames = _safe_load_frames(region, int(horizon))
    if not frames:
        return "", "", "", frame_state, "No PNG frames found. Run generate_predictions.py and generate_pngs.py first."

    i = int(frame_state.get("i", 0))
    i = i % len(frames)
    frame = frames[i]
    label = f"{region.upper()} | Horizon t+{horizon} | {frame['timestamp']}"
    actual_src = _img_data(os.path.join(PROJECT_ROOT, frame["actual_png"]))
    pred_src = _img_data(os.path.join(PROJECT_ROOT, frame["pred_png"]))

    if not play_state.get("paused", False):
        i = (i + 1) % len(frames)
    return actual_src, pred_src, label, {"i": i}, ""


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8051)
