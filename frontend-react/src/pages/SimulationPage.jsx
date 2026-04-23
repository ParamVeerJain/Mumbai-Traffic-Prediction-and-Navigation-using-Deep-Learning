import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import { CircleMarker, MapContainer, Polyline, TileLayer, Tooltip, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import CustomSelect from "../components/CustomSelect";
import { Activity, MapPinned, BrainCircuit, Zap, Play, Pause, Compass, Map as MapIcon, Waves } from "lucide-react";
import { motion } from "framer-motion";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function FitMapToRoute({ bounds }) {
  const map = useMap();
  useEffect(() => {
    setTimeout(() => {
      map.invalidateSize();
      if (bounds && bounds.length > 0) {
        map.fitBounds(bounds, { padding: [32, 32] });
      }
    }, 100);
  }, [map, bounds]);
  return null;
}

function SimulationMap({
  floodedAstarPath = [],
  sacPath = [],
  sourceMarker = null,
  destinationMarker = null,
  sacExplored = []
}) {
  const floodedAstarLines = useMemo(
    () => floodedAstarPath.map((seg) => (Array.isArray(seg.geometry) && seg.geometry.length > 1 ? seg.geometry : [[seg.u_lat, seg.u_lon], [seg.v_lat, seg.v_lon]])),
    [floodedAstarPath]
  );
  const sacLines = useMemo(
    () => sacPath.map((seg) => (Array.isArray(seg.geometry) && seg.geometry.length > 1 ? seg.geometry : [[seg.u_lat, seg.u_lon], [seg.v_lat, seg.v_lon]])),
    [sacPath]
  );

  const bounds = useMemo(() => {
    let pts = [...floodedAstarLines.flat(), ...sacLines.flat()];
    if (!pts.length && sacExplored.length) {
      pts = [...sacExplored].map((n) => [n.lat, n.lon]);
    }
    return pts.length ? pts : null;
  }, [floodedAstarLines, sacLines, sacExplored]);

  const sourcePoint = sourceMarker ? [sourceMarker.lat, sourceMarker.lon] : null;
  const destinationPoint = destinationMarker ? [destinationMarker.lat, destinationMarker.lon] : null;

  return (
    <div className="mapWrap demoMapWrap">
      {floodedAstarLines.length || sacLines.length || sacExplored.length ? (
        <MapContainer className="leafletMap" zoom={13} center={sourcePoint || [19.1663, 72.8526]} scrollWheelZoom>
          <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          <FitMapToRoute bounds={bounds} />

          {sacExplored.map((n, i) => (
            <CircleMarker
              key={`sac-explored-${i}`}
              center={[n.lat, n.lon]}
              radius={2}
              pathOptions={{ color: "#1e3a8a", fillColor: "#1e3a8a", fillOpacity: 0.5, weight: 0 }}
            />
          ))}

          {floodedAstarLines.map((positions, i) => (
            <Polyline key={`flooded-astar-${i}`} positions={positions} pathOptions={{ color: "#7f1d1d", weight: 6, opacity: 0.9, dashArray: "8,8" }} />
          ))}
          {sacLines.map((positions, i) => (
            <Polyline key={`sac-${i}`} positions={positions} pathOptions={{ color: "#1e3a8a", weight: 6, opacity: 0.95 }} />
          ))}

          {sourcePoint ? (
            <CircleMarker center={sourcePoint} radius={7} pathOptions={{ color: "#f59e0b", fillColor: "#f59e0b", fillOpacity: 1, weight: 2 }}>
              <Tooltip permanent direction="top" offset={[0, -8]} className="poiTooltip">
                {sourceMarker?.name || "Source"}
              </Tooltip>
            </CircleMarker>
          ) : null}
          {destinationPoint ? (
            <CircleMarker center={destinationPoint} radius={7} pathOptions={{ color: "#eab308", fillColor: "#eab308", fillOpacity: 1, weight: 2 }}>
              <Tooltip permanent direction="top" offset={[0, -8]} className="poiTooltip">
                {destinationMarker?.name || "Destination"}
              </Tooltip>
            </CircleMarker>
          ) : null}
        </MapContainer>
      ) : (
        <div className="mapEmpty">
          <MapPinned size={40} className="muted-icon" />
          <span>Simulation frames are not available.</span>
        </div>
      )}
    </div>
  );
}

export default function SimulationPage() {
  const [locations, setLocations] = useState([]);
  const [source, setSource] = useState("");
  const [destination, setDestination] = useState("");
  const [floodFactor, setFloodFactor] = useState(2.5);
  const [simResult, setSimResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [clockSec, setClockSec] = useState(0);

  useEffect(() => {
    axios
      .get(`${API_BASE}/locations`)
      .then((res) => {
        const locs = res.data.locations || [];
        setLocations(locs);
        setSource(locs[6] || locs[0] || "");
        setDestination(locs[3] || locs[1] || "");
      })
      .catch(() => setError("Could not load locations."));
  }, []);

  const loadSimulation = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await axios.post(`${API_BASE}/route/sac-simulation`, {
        source,
        destination,
        start_datetime: new Date().toISOString().slice(0, 16),
        algorithm: "astar",
        flood_ttr: Number(floodFactor)
      });
      setSimResult(res.data);
      setClockSec(0);
      setIsPlaying(false);
    } catch (e) {
      setSimResult(null);
      setError(e?.response?.data?.detail || "Unable to load simulation.");
    } finally {
      setLoading(false);
    }
  };

  const floodedPath = simResult?.flooded_astar_path || [];
  const sacFullPath = simResult?.sac_path || [];
  const hasPaths = floodedPath.length > 0 && sacFullPath.length > 0;

  const floodedEdgeSec = simResult?.flooded_astar_edge_seconds || [];
  const sacEdgeSec = simResult?.sac_edge_seconds || [];

  const floodedTotalSec = floodedEdgeSec.reduce((a, b) => a + Number(b || 0), 0);
  const sacTotalSec = sacEdgeSec.reduce((a, b) => a + Number(b || 0), 0);

  const floodedStep = useMemo(() => {
    let t = 0;
    let i = 0;
    for (; i < floodedEdgeSec.length; i += 1) {
      t += Number(floodedEdgeSec[i] || 0);
      if (t > clockSec) break;
    }
    return Math.min(i, floodedPath.length);
  }, [clockSec, floodedEdgeSec, floodedPath.length]);

  const sacStep = useMemo(() => {
    let t = 0;
    let i = 0;
    for (; i < sacEdgeSec.length; i += 1) {
      t += Number(sacEdgeSec[i] || 0);
      if (t > clockSec) break;
    }
    return Math.min(i, sacFullPath.length);
  }, [clockSec, sacEdgeSec, sacFullPath.length]);

  const floodedProgressPath = useMemo(() => floodedPath.slice(0, floodedStep), [floodedPath, floodedStep]);
  const sacProgressPath = useMemo(() => sacFullPath.slice(0, sacStep), [sacFullPath, sacStep]);

  useEffect(() => {
    if (!isPlaying || !hasPaths) return undefined;
    const intervalMs = 150;
    const maxTotal = Math.max(1, floodedTotalSec || 1, sacTotalSec || 1);
    const dt = Math.max(5, Math.round(maxTotal / 140)); // ~140 ticks per run (clamped)
    const id = setInterval(() => {
      setClockSec((prev) => {
        const next = prev + dt;
        if (next >= maxTotal) {
          setIsPlaying(false);
          return prev;
        }
        return next;
      });
    }, intervalMs);
    return () => clearInterval(id);
  }, [isPlaying, hasPaths, floodedTotalSec, sacTotalSec]);

  const locationOptions = useMemo(() => locations.map((loc) => ({ label: loc, value: loc })), [locations]);

  return (
    <motion.div className="app-page" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
      <header className="app-header" style={{ marginBottom: "20px" }}>
        <h1>
          A* vs SAC <span className="gradient-text">Simulation</span>
        </h1>
        <p>Fixed Monday 08:00, t+1 predictions. Playback speed adapts to SAC time saved.</p>
      </header>

      <div className="app-grid" style={{ gridTemplateColumns: "350px 1fr" }}>
        <motion.section className="glass-panel controls" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.5, delay: 0.2 }}>
          <h2>Scenario Settings</h2>
          <div className="input-group">
            <label>Source</label>
            <CustomSelect value={source} onChange={setSource} options={locationOptions} placeholder="Select source" icon={Compass} />
          </div>
          <div className="input-group">
            <label>Destination</label>
            <CustomSelect value={destination} onChange={setDestination} options={locationOptions} placeholder="Select destination" icon={MapIcon} />
          </div>
          <div className="input-group" style={{ marginTop: 18 }}>
            <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <Waves size={16} className="text-accent" /> Flood Severity Factor
              </span>
              <span className="badge-warning" style={{ padding: "4px 8px", borderRadius: "8px", fontSize: "1rem" }}>
                {floodFactor}x
              </span>
            </label>
            <input
              type="range"
              min="0"
              max="4"
              step="0.1"
              value={floodFactor}
              onChange={(e) => setFloodFactor(e.target.value)}
              className="styled-slider"
            />
            <div style={{ display: "flex", justifyContent: "space-between", color: "var(--text-secondary)", fontSize: "0.75rem", marginTop: "8px" }}>
              <span>Clear Traffic (0x)</span>
              <span>Severe Gridlock (4x)</span>
            </div>
          </div>

          <button className="btn btn-primary w-full mt-4" onClick={loadSimulation} disabled={loading || !source || !destination}>
            {loading ? "Loading..." : "Generate Simulation"}
          </button>

          <div className="input-group" style={{ marginTop: 24 }}>
            <label>Playback</label>
            <div style={{ display: "flex", gap: 10 }}>
              <button className="btn btn-primary" onClick={() => setIsPlaying((p) => !p)} disabled={!hasPaths} style={{ flex: 1 }}>
                {isPlaying ? (
                  <>
                    <Pause size={16} /> Pause
                  </>
                ) : (
                  <>
                    <Play size={16} /> Play
                  </>
                )}
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => {
                  setClockSec(0);
                  setIsPlaying(false);
                }}
                disabled={!hasPaths}
              >
                Reset
              </button>
            </div>
            <small className="muted">
              {hasPaths
                ? `Flooded A*: ${Math.min(floodedStep, floodedPath.length)} / ${floodedPath.length} • SAC: ${Math.min(sacStep, sacFullPath.length)} / ${sacFullPath.length}`
                : "Simulation paths are not available yet."}
            </small>
          </div>
          {error && <div className="error" style={{ marginTop: "16px", color: "#ef4444" }}>{error}</div>}
        </motion.section>

        <motion.section className="glass-panel map-section" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.5, delay: 0.3 }} style={{ display: "flex", flexDirection: "column" }}>
          {loading ? (
            <div className="mapEmpty" style={{ flex: 1 }}>
              <Activity size={40} className="muted-icon" />
              <span>Loading simulation...</span>
            </div>
          ) : simResult ? (
            <>
              <div className="demo-kpis" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "16px", marginBottom: "20px", marginTop: 0 }}>
                <div className="kpi" style={{ borderLeft: "4px solid #7f1d1d", alignItems: "flex-start" }}>
                  <div style={{ width: "100%" }}>
                    <span>Flooded A*</span>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginTop: "4px" }}>
                      <strong style={{ fontSize: "1.5rem" }}>{simResult.flooded_astar_time_min} min</strong>
                      <strong style={{ fontSize: "1.1rem", color: "#7f1d1d" }}>{simResult.flooded_astar_distance_km} km</strong>
                    </div>
                  </div>
                </div>
                <div className="kpi" style={{ borderLeft: "4px solid #1e3a8a", alignItems: "flex-start" }}>
                  <div style={{ width: "100%" }}>
                    <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                      <BrainCircuit size={12} /> SAC Agent
                    </span>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginTop: "4px" }}>
                      <strong style={{ fontSize: "1.5rem" }}>{simResult.sac_time_min} min</strong>
                      <strong style={{ fontSize: "1.1rem", color: "#1e3a8a" }}>{simResult.sac_distance_km} km</strong>
                    </div>
                  </div>
                </div>
                <div className="kpi" style={{ background: "rgba(139, 92, 246, 0.1)", borderColor: "var(--accent-primary)", alignItems: "center" }}>
                  <div>
                    <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                      <Zap size={12} className="text-accent" /> SAC Time Saved
                    </span>
                    <strong className="gradient-text" style={{ fontSize: "1.75rem" }}>
                      {simResult.sac_gain_vs_flooded_pct}%
                    </strong>
                  </div>
                </div>
              </div>

              <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", marginBottom: "12px", padding: "0 8px" }}>
                <div className="demo-distances" style={{ margin: 0, display: "flex", gap: "16px" }}>
                  <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <span style={{ width: "12px", height: "3px", background: "#7f1d1d", display: "inline-block" }}></span> A* Route
                  </span>
                  <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <span style={{ width: "12px", height: "3px", background: "#1e3a8a", display: "inline-block" }}></span> SAC Route
                  </span>
                </div>
              </div>

              <SimulationMap
                floodedAstarPath={floodedProgressPath}
                sacPath={sacProgressPath}
                sourceMarker={simResult.source_marker || null}
                destinationMarker={simResult.destination_marker || null}
                sacExplored={simResult.sac_explored_nodes || []}
              />
            </>
          ) : (
            <div className="mapEmpty" style={{ flex: 1 }}>
              <MapPinned size={40} className="muted-icon" />
              <span>Simulation unavailable.</span>
            </div>
          )}
        </motion.section>
      </div>
    </motion.div>
  );
}
