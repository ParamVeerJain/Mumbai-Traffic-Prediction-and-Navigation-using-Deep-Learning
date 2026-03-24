# ui/app.py
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from config import DATA_DIR, MODELS_DIR
from src.graph import load_graph, snap_to_node
from src.predictor import TrafficPredictor
from src.routing.astar import (astar_route, route_summary,
                                MODE_FASTEST, MODE_NO_SIGNALS, MODE_NO_TOLL,
                                _collect_edge_details)
from src.routing.ppo import PPOAgent
from src.utils import geocode, horizon_from_departure, format_duration, path_to_coords

# ── Load resources ────────────────────────────────────────────────────────────
GRAPH_CACHE = os.path.join(DATA_DIR, "graph.pkl")
IDX_PATH    = os.path.join(DATA_DIR, "road_to_idx.pkl")
CSV_PATH    = os.path.join(DATA_DIR, "dummy_history.csv")

print("[app] Loading graph...")
G = load_graph(cache_path=GRAPH_CACHE)

print("[app] Loading road index...")
with open(IDX_PATH, "rb") as f:
    road_to_idx = pickle.load(f)
num_roads = len(road_to_idx)

print("[app] Loading history CSV...")
df_history = (pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
              if os.path.exists(CSV_PATH) else None)

print("[app] Loading LSTM predictor...")
predictor = TrafficPredictor(road_to_idx, num_roads, df_history=df_history)

print("[app] Loading PPO agent...")
ppo_agent = PPOAgent()

nodes_list = list(G.nodes(data=True))
centre_lat = float(np.mean([d["lat"] for _, d in nodes_list]))
centre_lon = float(np.mean([d["lon"] for _, d in nodes_list]))

# ── Design tokens ─────────────────────────────────────────────────────────────
ACCENT   = "#00d4ff"
ACCENT2  = "#7b61ff"
BG_DEEP  = "#080a0f"
BG_CARD  = "#0f1117"
BG_INPUT = "#1a1d27"
BORDER   = "#2a2d3a"

CARD_STYLE = {
    "background": BG_CARD,
    "border": f"1px solid {BORDER}",
    "borderRadius": "12px",
    "padding": "20px",
}
LABEL_STYLE = {
    "color": "#888",
    "fontSize": "0.68rem",
    "letterSpacing": "0.13em",
    "fontFamily": "JetBrains Mono",
    "marginBottom": "4px",
    "display": "block",
}
INPUT_STYLE = {
    "background": BG_INPUT,
    "border": f"1px solid {BORDER}",
    "color": "#fff",
    "borderRadius": "8px",
    "marginBottom": "14px",
    "fontSize": "0.9rem",
}


# ── ✅ ALL helpers defined BEFORE app.layout ──────────────────────────────────

def _legend_dot(colour, label):
    return html.Div([
        html.Span(style={
            "display": "inline-block", "width": "10px", "height": "10px",
            "borderRadius": "50%", "background": colour, "marginRight": "8px",
        }),
        html.Span(label, style={
            "fontFamily": "Space Grotesk", "fontSize": "0.78rem", "color": "#aaa",
        }),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "5px"})


def _base_map():
    fig = go.Figure()
    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=centre_lat, lon=centre_lon),
            zoom=12,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=BG_DEEP,
        plot_bgcolor=BG_DEEP,
        showlegend=True,
        legend=dict(
            bgcolor=BG_CARD, bordercolor=BORDER, borderwidth=1,
            font=dict(color="#ccc", size=11, family="JetBrains Mono"),
            x=0.01, y=0.99,
        ),
        uirevision="map",
    )
    return fig


def _route_colour(mode):
    return {
        "fastest":    ACCENT,
        "no_signals": ACCENT2,
        "no_toll":    "#00ff88",
        "ppo":        "#ff9500",
    }.get(mode, ACCENT)


def _congestion_colour(sr):
    if sr > 0.70: return "#00cc66"
    if sr > 0.45: return "#ffaa00"
    return "#ff3344"


def _metric_card(label, value, unit=""):
    return html.Div([
        html.P(label, style={
            "color": "#555", "fontSize": "0.62rem", "fontFamily": "JetBrains Mono",
            "margin": "0 0 2px", "letterSpacing": "0.1em",
        }),
        html.P(f"{value}{unit}", style={
            "color": "#fff", "fontSize": "1.25rem",
            "fontFamily": "Space Grotesk", "fontWeight": "600", "margin": "0",
        }),
    ], style={
        "background": BG_DEEP, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "10px 14px",
    })


def _metrics_cards(summary):
    return html.Div([
        dbc.Row([
            dbc.Col(_metric_card("DISTANCE",    summary["distance_km"],   " km"),  width=6),
            dbc.Col(_metric_card("TRAVEL TIME", summary["travel_min"],    " min"), width=6),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(_metric_card("AVG SPEED", summary["avg_speed_kmh"], " km/h"), width=6),
            dbc.Col(_metric_card("SIGNALS",   summary["signal_roads"]),           width=3),
            dbc.Col(_metric_card("TOLLS",     summary["toll_roads"]),             width=3),
        ], className="g-2", style={"marginTop": "8px"}),
    ])


# ── App init ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700"
        "&family=JetBrains+Mono:wght@400;500&display=swap",
    ],
    title="Urban Traffic Router",
)

app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        * { box-sizing: border-box; }
        body { background: #080a0f !important; margin: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0f1117; }
        ::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 2px; }
        .form-control:focus {
            background: #1a1d27 !important;
            border-color: #00d4ff !important;
            color: #fff !important;
            box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
        }
        .form-control { color: #fff !important; }
        .form-check-input:checked { background-color: #00d4ff; border-color: #00d4ff; }
        .mode-radio .form-check { padding: 8px 10px; border-radius: 6px; transition: background .15s; }
        .mode-radio .form-check:hover { background: rgba(255,255,255,0.04); }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""

# ── Layout (defined AFTER all helpers) ───────────────────────────────────────
app.layout = dbc.Container(
    fluid=True,
    style={"background": BG_DEEP, "minHeight": "100vh"},
    children=[

        dbc.Row(dbc.Col(html.Div([
            html.Div([
                html.Span("URBAN",   style={"color": ACCENT,  "fontWeight": "700"}),
                html.Span(" TRAFFIC ", style={"color": "#fff", "fontWeight": "700"}),
                html.Span("ROUTER", style={"color": ACCENT2, "fontWeight": "700"}),
            ], style={"fontFamily": "Space Grotesk", "fontSize": "1.9rem",
                      "letterSpacing": "0.12em", "lineHeight": "1"}),
            html.P("LSTM · A* · PPO  |  OpenStreetMap  |  POC",
                   style={"color": "#444", "fontFamily": "JetBrains Mono",
                          "fontSize": "0.68rem", "letterSpacing": "0.2em",
                          "margin": "6px 0 0"}),
        ], style={"padding": "22px 0 14px"}))),

        dbc.Row([

            dbc.Col(width=3, children=[
                html.Div(style=CARD_STYLE, children=[

                    html.Span("ORIGIN", style=LABEL_STYLE),
                    dbc.Input(id="origin-input",
                              placeholder="e.g. Andheri, Mumbai",
                              value="Andheri, Mumbai",
                              style=INPUT_STYLE),

                    html.Span("DESTINATION", style=LABEL_STYLE),
                    dbc.Input(id="dest-input",
                              placeholder="e.g. Borivali, Mumbai",
                              value="Borivali, Mumbai",
                              style=INPUT_STYLE),

                    html.Span("DEPARTURE TIME", style=LABEL_STYLE),
                    dbc.Input(id="time-input",
                              type="datetime-local",
                              value=pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M"),
                              style=INPUT_STYLE),

                    html.Hr(style={"borderColor": BORDER, "margin": "6px 0 14px"}),

                    html.Span("ROUTING MODE", style=LABEL_STYLE),
                    dbc.RadioItems(
                        id="mode-select",
                        className="mode-radio",
                        options=[
                            {"label": html.Span("🚀  Fastest route",
                                                style={"fontFamily": "Space Grotesk",
                                                       "fontSize": "0.85rem"}),
                             "value": "fastest"},
                            {"label": html.Span("🟢  Avoid traffic signals",
                                                style={"fontFamily": "Space Grotesk",
                                                       "fontSize": "0.85rem"}),
                             "value": "no_signals"},
                            {"label": html.Span("💳  Avoid toll roads",
                                                style={"fontFamily": "Space Grotesk",
                                                       "fontSize": "0.85rem"}),
                             "value": "no_toll"},
                            {"label": html.Span("🤖  PPO learned route",
                                                style={"fontFamily": "Space Grotesk",
                                                       "fontSize": "0.85rem"}),
                             "value": "ppo"},
                        ],
                        value="fastest",
                        inline=False,
                        style={"marginBottom": "20px", "color": "#ccc"},
                    ),

                    dbc.Button(
                        "FIND ROUTE  →",
                        id="route-btn", n_clicks=0,
                        style={
                            "background": f"linear-gradient(135deg, {ACCENT}, {ACCENT2})",
                            "border": "none", "width": "100%",
                            "fontFamily": "Space Grotesk", "fontWeight": "600",
                            "letterSpacing": "0.12em", "padding": "12px 0",
                            "borderRadius": "8px", "fontSize": "0.9rem",
                            "cursor": "pointer", "color": "#000",
                        },
                    ),

                    html.Div(id="status-msg",
                             style={"marginTop": "10px",
                                    "fontFamily": "JetBrains Mono",
                                    "fontSize": "0.72rem",
                                    "color": "#666", "minHeight": "20px"}),
                ]),

                html.Div(id="horizon-panel", style={"marginTop": "10px"}),
                html.Div(id="metrics-panel", style={"marginTop": "10px"}),

                html.Div(style={**CARD_STYLE, "padding": "12px 16px",
                                "marginTop": "10px"}, children=[
                    html.Span("CONGESTION LEGEND", style=LABEL_STYLE),
                    _legend_dot("#00cc66", "Low  (>70% speed)"),
                    _legend_dot("#ffaa00", "Medium  (45–70%)"),
                    _legend_dot("#ff3344", "High  (<45%)"),
                ]),
            ]),

            dbc.Col(width=9, children=[
                dcc.Loading(
                    id="map-loading",
                    type="circle",
                    color=ACCENT,
                    children=dcc.Graph(
                        id="map-graph",
                        figure=_base_map(),          # ✅ initial figure
                        style={"height": "85vh", "borderRadius": "12px",
                               "overflow": "hidden",
                               "border": f"1px solid {BORDER}"},
                        config={"scrollZoom": True, "displayModeBar": False},
                    )
                )
            ]),
        ]),

        html.Div(style={"height": "24px"}),
    ]
)


# ── Callback ──────────────────────────────────────────────────────────────────
@app.callback(
    Output("map-graph",     "figure"),
    Output("status-msg",    "children"),
    Output("metrics-panel", "children"),
    Output("horizon-panel", "children"),
    Input("route-btn",      "n_clicks"),
    State("origin-input",   "value"),
    State("dest-input",     "value"),
    State("time-input",     "value"),
    State("mode-select",    "value"),
    prevent_initial_call=True,        # ✅ don't fire on page load
)
def find_route(n_clicks, origin_str, dest_str, time_str, mode):
    empty = html.Div()

    try:
        all_nodes = list(G.nodes(data=True))

        try:
            olat, olon = geocode(origin_str)
        except Exception:
            olat, olon = all_nodes[0][1]["lat"], all_nodes[0][1]["lon"]

        try:
            dlat, dlon = geocode(dest_str)
        except Exception:
            dlat, dlon = all_nodes[-1][1]["lat"], all_nodes[-1][1]["lon"]

        origin_node = snap_to_node(G, olat, olon)
        dest_node   = snap_to_node(G, dlat, dlon)

        if origin_node == dest_node:
            return _base_map(), "⚠ Origin and destination map to the same node.", empty, empty

        try:
            departure_dt = pd.Timestamp(time_str)
        except Exception:
            departure_dt = pd.Timestamp.now()

        horizon_step = horizon_from_departure(departure_dt)
        predictor.predict_graph(G, departure_dt, horizon_step)

        if mode == "ppo":
            path = ppo_agent.route(G, origin_node, dest_node,
                                   departure_hour=departure_dt.hour, mode="fastest")
            if path and len(path) >= 2:
                edge_details = _collect_edge_details(G, path)
                status = f"🤖 PPO route  |  horizon t+{horizon_step}"
            else:
                path, _, edge_details = astar_route(G, origin_node, dest_node, MODE_FASTEST)
                status = f"⚠ PPO → A* fallback  |  horizon t+{horizon_step}"
        else:
            mode_map = {"fastest": MODE_FASTEST,
                        "no_signals": MODE_NO_SIGNALS,
                        "no_toll": MODE_NO_TOLL}
            path, _, edge_details = astar_route(
                G, origin_node, dest_node, mode_map.get(mode, MODE_FASTEST))
            labels = {"fastest": "🚀 Fastest", "no_signals": "🟢 No signals",
                      "no_toll": "💳 No toll"}
            status = f"{labels.get(mode, 'A*')}  |  horizon t+{horizon_step}"

        if not path or len(path) < 2:
            return _base_map(), "❌ No route found.", empty, empty

        summary = route_summary(edge_details)
        fig     = _base_map()
        coords  = path_to_coords(G, path)
        lats    = [c[0] for c in coords]
        lons    = [c[1] for c in coords]

        # Congestion-coloured segments
        for ed in edge_details:
            sr = ed.get("speed_ratio", 1.0)
            fig.add_trace(go.Scattermap(
                lat=[ed["u_lat"], ed["v_lat"]],
                lon=[ed["u_lon"], ed["v_lon"]],
                mode="lines",
                line=dict(width=4, color=_congestion_colour(sr)),
                showlegend=False,
                hovertemplate=(
                    f"<b>{ed['highway']}</b><br>"
                    f"Speed: {ed['predicted_speed']:.1f} km/h "
                    f"(ff: {ed['free_flow_speed']:.0f})<br>"
                    f"Ratio: {sr:.2f} | TT: {format_duration(ed['travel_time_s'])}<br>"
                    f"Length: {ed['length_m']:.0f} m"
                    + ("<br>🚦 Signals" if ed["traffic_signals"] else "")
                    + ("<br>💳 Toll"    if ed["toll"] else "")
                    + "<extra></extra>"
                ),
            ))

        # Route outline
        fig.add_trace(go.Scattermap(
            lat=lats, lon=lons, mode="lines",
            line=dict(width=7, color=_route_colour(mode)),
            opacity=0.35,
            name=f"Route ({mode})", hoverinfo="skip",
        ))

        # Markers
        for lat, lon, colour, label in [
            (lats[0],  lons[0],  "#00ff88", "  Origin"),
            (lats[-1], lons[-1], "#ff4466", "  Destination"),
        ]:
            fig.add_trace(go.Scattermap(
                lat=[lat], lon=[lon],
                mode="markers+text",
                marker=dict(size=16, color=colour, symbol="circle"),
                text=[label],
                textposition="middle right",
                textfont=dict(color=colour, size=12, family="Space Grotesk"),
                name=label.strip(),
                hovertemplate=f"{label.strip()}<br>{lat:.5f}, {lon:.5f}<extra></extra>",
            ))

        # Re-centre
        lat_span = max(abs(max(lats) - min(lats)), 0.005)
        lon_span = max(abs(max(lons) - min(lons)), 0.005)
        zoom = max(9, min(15, int(12 - np.log2(max(lat_span, lon_span) * 40))))
        fig.update_layout(map=dict(
            style="open-street-map",
            center=dict(lat=(lats[0]+lats[-1])/2, lon=(lons[0]+lons[-1])/2),
            zoom=zoom,
        ))

        horizon_div = html.Div([
            html.Span("⏱ ", style={"fontSize": "1rem"}),
            html.Span(f"LSTM HORIZON  t+{horizon_step}",
                      style={"fontFamily": "JetBrains Mono", "fontSize": "0.72rem",
                             "letterSpacing": "0.1em", "color": ACCENT}),
            html.Span(f"  ({horizon_step} hr{'s' if horizon_step > 1 else ''} ahead)",
                      style={"fontFamily": "JetBrains Mono",
                             "fontSize": "0.68rem", "color": "#555"}),
        ], style={**CARD_STYLE, "padding": "10px 16px"})

        return fig, status, _metrics_cards(summary), horizon_div

    except Exception as e:
        import traceback
        traceback.print_exc()
        return _base_map(), f"❌ Error: {e}", empty, empty


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)