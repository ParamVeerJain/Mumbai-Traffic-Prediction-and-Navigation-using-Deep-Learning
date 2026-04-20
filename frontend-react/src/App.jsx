import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  CircleMarker,
  MapContainer,
  Polyline,
  TileLayer,
  Tooltip,
  useMap
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { Gauge, MapPinned, Navigation, Route, Timer } from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function FitMapToRoute({ bounds }) {
  const map = useMap();

  useEffect(() => {
    if (!bounds) return;
    map.fitBounds(bounds, { padding: [32, 32] });
  }, [map, bounds]);

  return null;
}

function MapView({ path = [], sourceMarker = null, destinationMarker = null }) {
  const segmentLines = useMemo(() => {
    return path.map((seg) => {
      if (Array.isArray(seg.geometry) && seg.geometry.length > 1) {
        return seg.geometry;
      }
      return [
        [seg.u_lat, seg.u_lon],
        [seg.v_lat, seg.v_lon]
      ];
    });
  }, [path]);

  const bounds = useMemo(() => {
    if (!segmentLines.length) return null;
    const pts = segmentLines.flat();
    return pts;
  }, [segmentLines]);

  const sourcePoint = sourceMarker ? [sourceMarker.lat, sourceMarker.lon] : path.length ? [path[0].u_lat, path[0].u_lon] : null;
  const destinationPoint = destinationMarker
    ? [destinationMarker.lat, destinationMarker.lon]
    : path.length
      ? [path[path.length - 1].v_lat, path[path.length - 1].v_lon]
      : null;

  return (
    <div className="mapWrap">
      {path.length ? (
        <MapContainer className="leafletMap" zoom={13} center={sourcePoint || [19.1663, 72.8526]} scrollWheelZoom>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <FitMapToRoute bounds={bounds} />
          {segmentLines.map((positions, i) => (
            <Polyline key={`route-main-${i}`} positions={positions} pathOptions={{ color: "#22d3ee", weight: 6, opacity: 0.92 }} />
          ))}

          {path.map((seg, i) => (
            <Polyline
              key={`${seg.road_name}-${seg.edge_key}-${i}`}
              positions={segmentLines[i]}
              pathOptions={{ color: "#38bdf8", weight: 10, opacity: 0 }}
            >
              <Tooltip sticky>
                {seg.road_name}
                <br />
                Near {seg.near_area}
              </Tooltip>
            </Polyline>
          ))}

          {sourcePoint ? (
            <CircleMarker
              center={sourcePoint}
              radius={8}
              pathOptions={{ color: "#f59e0b", fillColor: "#f59e0b", fillOpacity: 1, weight: 2 }}
            >
              <Tooltip permanent direction="top" offset={[0, -8]} className="poiTooltip">
                {sourceMarker?.name || "Source"}
              </Tooltip>
            </CircleMarker>
          ) : null}

          {destinationPoint ? (
            <CircleMarker
              center={destinationPoint}
              radius={8}
              pathOptions={{ color: "#ef4444", fillColor: "#ef4444", fillOpacity: 1, weight: 2 }}
            >
              <Tooltip permanent direction="top" offset={[0, -8]} className="poiTooltip">
                {destinationMarker?.name || "Destination"}
              </Tooltip>
            </CircleMarker>
          ) : null}
        </MapContainer>
      ) : null}
      {!path.length && <div className="mapEmpty">Select route and click Find Best Route</div>}
    </div>
  );
}

export default function App() {
  const [locations, setLocations] = useState([]);
  const [source, setSource] = useState("");
  const [destination, setDestination] = useState("");
  const [startDatetime, setStartDatetime] = useState("2024-07-01T08:00");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    axios
      .get(`${API_BASE}/locations`)
      .then((res) => {
        const locs = res.data.locations || [];
        setLocations(locs);
        setSource(locs[6] || locs[0] || "");
        setDestination(locs[3] || locs[1] || "");
      })
      .catch(() => setError("Could not load locations. Is API running on port 8000?"));
  }, []);

  const runRoute = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await axios.post(`${API_BASE}/route`, {
        source,
        destination,
        start_datetime: startDatetime
      });
      setResult(res.data);
    } catch (e) {
      setResult(null);
      setError(e?.response?.data?.detail || "Unable to find route right now.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="top">
        <h1>Mumbai Smart Navigation</h1>
        <p>Beautiful route intelligence for Goregaon with predicted traffic delays.</p>
      </header>

      <div className="grid">
        <section className="card controls">
          <h2>Trip Planner</h2>
          <label>Source</label>
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            {locations.map((loc) => (
              <option key={loc} value={loc}>
                {loc}
              </option>
            ))}
          </select>

          <label>Destination</label>
          <select value={destination} onChange={(e) => setDestination(e.target.value)}>
            {locations.map((loc) => (
              <option key={loc} value={loc}>
                {loc}
              </option>
            ))}
          </select>

          <label>Start Date & Time</label>
          <input type="datetime-local" value={startDatetime} onChange={(e) => setStartDatetime(e.target.value)} />

          <button onClick={runRoute} disabled={loading || !source || !destination}>
            {loading ? "Finding best route..." : "Find Best Route"}
          </button>

          {error && <div className="error">{error}</div>}
        </section>

        <section className="card mapCard">
          <div className="kpis">
            <div className="kpi">
              <Timer size={18} />
              <span>Est. Time</span>
              <strong>{result ? `${result.est_time_min} min` : "--"}</strong>
            </div>
            <div className="kpi">
              <Navigation size={18} />
              <span>Est. Distance</span>
              <strong>{result ? `${result.est_distance_km} km` : "--"}</strong>
            </div>
            <div className="kpi">
              <Route size={18} />
              <span>Segments</span>
              <strong>{result ? result.segments_used : "--"}</strong>
            </div>
          </div>
          <MapView
            path={result?.path || []}
            sourceMarker={result?.source_marker || null}
            destinationMarker={result?.destination_marker || null}
          />
        </section>
      </div>

      <section className="card delayCard">
        <h2>
          <Gauge size={18} /> Roads Causing Most Delay
        </h2>
        {!result?.delays?.length ? (
          <p className="muted">No route yet. Run Smart Navigation to see delay insights.</p>
        ) : (
          <div className="delayList">
            {result.delays.map((d, idx) => (
              <article key={`${d.road_name}-${idx}`} className="delayItem">
                <h3>{d.road_name}</h3>
                <p>
                  <MapPinned size={14} /> Near {d.near_area}
                </p>
                <p>Delay: {d.delay_seconds.toFixed(0)} sec</p>
                <p>Segment length: {d.segment_length_m.toFixed(0)} m</p>
                <p>Traffic factor: {d.traffic_factor.toFixed(2)}x</p>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
