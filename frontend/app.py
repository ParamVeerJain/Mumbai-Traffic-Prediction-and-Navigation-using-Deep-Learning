import base64
import json
import os
import sys
from functools import lru_cache
from typing import Dict, List

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, dcc, html
import plotly.graph_objects as go


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# DATA_ROOT = os.path.join(PROJECT_ROOT, "backend", "data generation", "data")
DATA_ROOT = r'E:\Projects\Major 2.0\Mumbai-Traffic-Prediction-and-Navigation-using-Deep-Learning\backend\data generation\data'
PNG_ROOT = os.path.join(DATA_ROOT, "pngs")
PRED_ROOT = os.path.join(DATA_ROOT, "predictions")
BACKEND_DATA_GEN_DIR = os.path.join(PROJECT_ROOT, "backend", "data generation")

if BACKEND_DATA_GEN_DIR not in sys.path:
    sys.path.append(BACKEND_DATA_GEN_DIR)

import tda_router_goregaon as tda


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
    return [1, 2, 3]


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

@lru_cache(maxsize=1)
def _router_assets():
    g, ctx, _ = tda.load_goregaon_graph_and_context(gwn_path=None)
    node_ids = list(ctx.node_coords.keys())
    node_xy = np.array([ctx.node_coords[n] for n in node_ids], dtype=np.float64)
    named_nodes = {}
    for name, lat, lon in tda.NAMED_LOCATIONS:
        named_nodes[name] = tda.nearest_node(lat, lon, node_ids, node_xy)
    return g, ctx, named_nodes


def _route_options():
    names = [nm for nm, _, _ in tda.NAMED_LOCATIONS]
    return [{"label": n, "value": n} for n in names]


def _base_router_figure(g):
    fig = go.Figure()
    for u, v in g.edges():
        fig.add_trace(
            go.Scatter(
                x=[g.nodes[u]["x"], g.nodes[v]["x"]],
                y=[g.nodes[u]["y"], g.nodes[v]["y"]],
                mode="lines",
                line={"color": "rgba(180,180,180,0.25)", "width": 1},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0b1020",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=620,
        title="Goregaon Smart Navigation Map",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
    )
    return fig


def _edge_road_name(g, u, v, k) -> str:
    data = g.get_edge_data(u, v, k) or {}
    name = data.get("name")
    if isinstance(name, list):
        name = ", ".join([str(x) for x in name if x])
    if not name:
        return "Unnamed road"
    return str(name)


def _nearest_area_name(g, named_nodes, x, y) -> str:
    nearest_name = "Nearby area"
    best_d2 = float("inf")
    for name, node_id in named_nodes.items():
        nx = g.nodes[node_id]["x"]
        ny = g.nodes[node_id]["y"]
        d2 = (nx - x) ** 2 + (ny - y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            nearest_name = name
    return nearest_name


def _datetime_to_hour_of_week(dt_str: str) -> int:
    """Convert user date-time to model hour-of-week (0..167)."""
    anchor = pd.Timestamp("2024-07-01 00:00:00")
    if not dt_str:
        return 8
    ts = pd.Timestamp(dt_str)
    delta_hours = int((ts - anchor).total_seconds() // 3600)
    return int(delta_hours % 168)


router_opts = _route_options()
default_src = "Goregaon Railway Station"
default_dst = "Film City"

app.layout = dbc.Container(
    fluid=True,
    style={
        "padding": "16px",
        "background": "linear-gradient(135deg, #030712 0%, #0b1020 45%, #111827 100%)",
        "minHeight": "100vh",
    },
    children=[
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "14px"},
            children=[
                html.Img(src=_logo(), style={"height": "48px"}),
                html.Div(
                    [
                        html.H2("Mumbai Traffic Pulse", style={"margin": 0, "fontWeight": 700}),
                        html.Div("Smart Congestion Simulator + Smart Navigation", style={"color": "#93c5fd"}),
                    ]
                ),
            ],
        ),
        html.Hr(),
        dcc.Tabs(
            id="main-tabs",
            value="sim",
            colors={"border": "#1f2937", "primary": "#38bdf8", "background": "#111827"},
            children=[
                dcc.Tab(
                    label="Simulator",
                    value="sim",
                    children=[
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
                ),
                dcc.Tab(
                    label="Smart Navigation",
                    value="router",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H4("Route Controls", className="card-title"),
                                                html.Label("Source"),
                                                dcc.Dropdown(id="src-loc", options=router_opts, value=default_src, clearable=False),
                                                html.Br(),
                                                html.Label("Destination"),
                                                dcc.Dropdown(id="dst-loc", options=router_opts, value=default_dst, clearable=False),
                                                html.Br(),
                                                html.Label("Trip Start Date & Time"),
                                                dcc.Input(
                                                    id="start-datetime",
                                                    type="datetime-local",
                                                    value="2024-07-01T08:00",
                                                    className="form-control",
                                                ),
                                                html.Small(
                                                    "Pick when the trip starts. The router uses this to estimate traffic.",
                                                    style={"color": "#9ca3af"},
                                                ),
                                                html.Br(),
                                                dbc.Button("Find Best Route", id="run-router", color="info", className="w-100"),
                                                dcc.Markdown(
                                                    "Choose source, destination, date-time, then click Find Best Route.",
                                                    id="router-summary",
                                                    style={"marginTop": "12px", "whiteSpace": "pre-wrap", "color": "#e5e7eb"},
                                                ),
                                            ]
                                        ),
                                        style={"backgroundColor": "#111827", "border": "1px solid #1f2937"},
                                    ),
                                    md=3,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(id="router-graph", figure=go.Figure()),
                                            ]
                                        ),
                                        style={"backgroundColor": "#111827", "border": "1px solid #1f2937"},
                                    ),
                                    md=9,
                                ),
                            ],
                            style={"marginTop": "12px"},
                        ),
                    ],
                ),
            ],
        ),
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


@app.callback(
    Output("router-graph", "figure"),
    Output("router-summary", "children"),
    Input("run-router", "n_clicks"),
    State("src-loc", "value"),
    State("dst-loc", "value"),
    State("start-datetime", "value"),
    prevent_initial_call=True,
)
def update_router(_, src_name, dst_name, start_dt):
    g, ctx, named_nodes = _router_assets()
    fig = _base_router_figure(g)

    if not src_name or not dst_name:
        return fig, "Choose both source and destination."
    if src_name == dst_name:
        return fig, "Source and destination must be different."

    src = named_nodes[src_name]
    dst = named_nodes[dst_name]
    start_hour = _datetime_to_hour_of_week(start_dt)
    path, total_sec = tda.tda_star(g, ctx, src, dst, start_hour=start_hour, start_elapsed_sec=0.0)
    if path is None:
        fig.add_annotation(text="No route found in bounded graph.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return (
            fig,
            f"Trip: {src_name} -> {dst_name}\n"
            f"Start time: {start_dt or 'not set'}\n\n"
            f"No route is available between these points inside the current Goregaon map boundary.",
        )

    for u, v, k in path:
        road_name = _edge_road_name(g, u, v, k)
        fig.add_trace(
            go.Scatter(
                x=[g.nodes[u]["x"], g.nodes[v]["x"]],
                y=[g.nodes[u]["y"], g.nodes[v]["y"]],
                mode="lines",
                line={"color": "#22d3ee", "width": 3},
                hovertemplate=f"Road: {road_name}<extra></extra>",
                showlegend=False,
            )
        )

    # Clean labels like goregaon_tda_routes.png style: show area names only.
    for area_name, node_id in named_nodes.items():
        fig.add_trace(
            go.Scatter(
                x=[g.nodes[node_id]["x"]],
                y=[g.nodes[node_id]["y"]],
                mode="text",
                text=[area_name],
                textfont={"size": 9, "color": "#93c5fd", "family": "Arial"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[g.nodes[src]["x"]],
            y=[g.nodes[src]["y"]],
            mode="markers+text",
            marker={"size": 11, "color": "#f59e0b", "symbol": "circle"},
            text=["Source"],
            textposition="top center",
            name="Source",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[g.nodes[dst]["x"]],
            y=[g.nodes[dst]["y"]],
            mode="markers+text",
            marker={"size": 14, "color": "#ef4444", "symbol": "star"},
            text=["Destination"],
            textposition="top center",
            name="Destination",
        )
    )

    edge_rows = tda.edge_stats_for_path(path, ctx, start_hour, 0.0)
    total_distance_km = sum(row[1] for row in edge_rows) / 1000.0
    edge_rows_named = []
    for (u, v, k), row in zip(path, edge_rows):
        mid_x = (g.nodes[u]["x"] + g.nodes[v]["x"]) / 2.0
        mid_y = (g.nodes[u]["y"] + g.nodes[v]["y"]) / 2.0
        near_area = _nearest_area_name(g, named_nodes, mid_x, mid_y)
        edge_rows_named.append((_edge_road_name(g, u, v, k), near_area, row))
    top_edges = sorted(edge_rows_named, key=lambda x: x[2][4], reverse=True)[:6]
    bottleneck_text = "\n".join(
        [
            (
                f"- {road_name} (near {near_area}): adds about {w:.0f} extra seconds "
                f"(segment length {length_m:.0f} m, traffic factor {ttr:.2f}x)"
            )
            for road_name, near_area, (_, length_m, _, ttr, w) in top_edges
        ]
    )
    summary = (
        f"Trip: {src_name} -> {dst_name}\n"
        f"Start time: {start_dt}\n"
        f"Est. time: {total_sec / 60.0:.1f} min  |  Est. distance: {total_distance_km:.2f} km\n"
        f"Road segments used: {len(path)}\n\n"
        f"Road names causing most delay:\n"
        f"{bottleneck_text}\n\n"
        f"What this means: these are the parts of the journey where traffic is expected to be heaviest."
    )
    return fig, summary


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8051)
