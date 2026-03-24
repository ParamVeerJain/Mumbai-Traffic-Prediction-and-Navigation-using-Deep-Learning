# src/utils.py
import math
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import time

_geocoder = Nominatim(user_agent="traffic_router_poc")


def geocode(address: str):
    """Returns (lat, lon) or raises."""
    try:
        loc = _geocoder.geocode(address, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except Exception as e:
        pass
    raise ValueError(f"Could not geocode: {address!r}")


def horizon_from_departure(departure_dt: pd.Timestamp,
                           now: pd.Timestamp = None) -> int:
    """
    How many hours until departure?
    Returns clamped int in [1, 9].
    departure in the past or < 30 min away → step 1 (next hour).
    """
    if now is None:
        now = pd.Timestamp.now()
    delta_h = (departure_dt - now).total_seconds() / 3600
    step    = max(1, min(9, math.ceil(delta_h)))
    return step


def format_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m >= 60:
        return f"{m//60}h {m%60}m"
    return f"{m}m {s}s"


def path_to_coords(G, path):
    """List of (lat, lon) for a path."""
    return [(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in path]