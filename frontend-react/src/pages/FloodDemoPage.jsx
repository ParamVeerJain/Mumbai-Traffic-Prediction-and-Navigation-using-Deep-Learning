import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import { CircleMarker, MapContainer, Polyline, TileLayer, Tooltip, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import CustomSelect from "../components/CustomSelect";
import { Activity, MapPinned, Waves, BrainCircuit, Zap, Compass, Map as MapIcon } from "lucide-react";
import { motion } from "framer-motion";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const DEFAULT_SOURCE = "Goregaon Railway Station";
const DEFAULT_DESTINATION = "Film City";

function getLocalDatetimeValue(date = new Date()) {
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 16);
}

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

function FloodDemoMap({
  floodedPath = [],
  sacPath = [],
  sourceMarker = null,
  destinationMarker = null,
  floodedExplored = [],
  sacExplored = []
}) {
  const floodedLines = useMemo(() =>
    floodedPath.map((seg) => Array.isArray(seg.geometry) && seg.geometry.length > 1 ? seg.geometry : [[seg.u_lat, seg.u_lon], [seg.v_lat, seg.v_lon]]),
    [floodedPath]
  );
  const sacLines = useMemo(() =>
    sacPath.map((seg) => Array.isArray(seg.geometry) && seg.geometry.length > 1 ? seg.geometry : [[seg.u_lat, seg.u_lon], [seg.v_lat, seg.v_lon]]),
    [sacPath]
  );

  const bounds = useMemo(() => {
    let pts = [...floodedLines.flat(), ...sacLines.flat()];
    if (!pts.length && (floodedExplored.length || sacExplored.length)) {
       pts = [...floodedExplored, ...sacExplored].map(n => [n.lat, n.lon]);
    }
    return pts.length ? pts : null;
  }, [floodedLines, sacLines, floodedExplored, sacExplored]);

  const sourcePoint = sourceMarker ? [sourceMarker.lat, sourceMarker.lon] : null;
  const destinationPoint = destinationMarker ? [destinationMarker.lat, destinationMarker.lon] : null;

  return (
    <div className="mapWrap demoMapWrap">
      {floodedLines.length || sacLines.length || floodedExplored.length ? (
        <MapContainer className="leafletMap" zoom={13} center={sourcePoint || [19.1663, 72.8526]} scrollWheelZoom>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <FitMapToRoute bounds={bounds} />
          
          {floodedExplored.map((n, i) => (
            <CircleMarker
              key={`flood-explored-${i}`}
              center={[n.lat, n.lon]}
              radius={2}
              pathOptions={{ color: "#7f1d1d", fillColor: "#7f1d1d", fillOpacity: 0.18, weight: 0 }}
            />
          ))}

          {sacExplored.map((n, i) => (
            <CircleMarker
              key={`sac-explored-${i}`}
              center={[n.lat, n.lon]}
              radius={2}
              pathOptions={{ color: "#1e3a8a", fillColor: "#1e3a8a", fillOpacity: 0.5, weight: 0 }}
            />
          ))}

          {floodedLines.map((positions, i) => (
            <Polyline key={`flood-${i}`} positions={positions} pathOptions={{ color: "#7f1d1d", weight: 6, opacity: 0.9, dashArray: "8,8" }} />
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
          <MapPinned size={40} className="muted-icon"/>
          <span>Select a route and run the demo</span>
        </div>
      )}
    </div>
  );
}

export default function FloodDemoPage() {
  const [locations, setLocations] = useState([]);
  const [source, setSource] = useState("");
  const [destination, setDestination] = useState("");
  const [floodFactor, setFloodFactor] = useState(1.5);
  const [demoResult, setDemoResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    axios.get(`${API_BASE}/locations`)
      .then((res) => {
        const locs = res.data.locations || [];
        setLocations(locs);
        setSource(locs.includes(DEFAULT_SOURCE) ? DEFAULT_SOURCE : locs[0] || "");
        setDestination(
          locs.includes(DEFAULT_DESTINATION) ? DEFAULT_DESTINATION : locs[1] || locs[0] || ""
        );
      })
      .catch(() => setError("Could not load locations."));
  }, []);

  const runDemo = async () => {
    setLoading(true);
    setError("");
    try {
      const startDatetime = getLocalDatetimeValue();
      const res = await axios.post(`${API_BASE}/route/flood-demo`, {
        source,
        destination,
        start_datetime: startDatetime,
        algorithm: "astar",
        flood_ttr: Number(floodFactor)
      });
      setDemoResult(res.data);
    } catch (e) {
      setDemoResult(null);
      setError(e?.response?.data?.detail || "Unable to run flood shock demo right now.");
    } finally {
      setLoading(false);
    }
  };

  const locationOptions = useMemo(() => locations.map(loc => ({ label: loc, value: loc })), [locations]);

  return (
    <motion.div 
      className="app-page"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <header className="app-header" style={{ marginBottom: "20px" }}>
        <h1>Flood Shock <span className="gradient-text">Simulator</span></h1>
        <p>Demonstrate SAC Deep RL's ability to seamlessly route around dynamically induced extreme congestion.</p>
      </header>

      <div className="app-grid" style={{ gridTemplateColumns: "350px 1fr" }}>
        <motion.section 
          className="glass-panel controls"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <h2>Scenario Settings</h2>
          
          <div className="input-group">
             <label>Source</label>
             <CustomSelect value={source} onChange={setSource} options={locationOptions} placeholder="Select source" icon={Compass} />
          </div>

          <div className="input-group">
             <label>Destination</label>
             <CustomSelect value={destination} onChange={setDestination} options={locationOptions} placeholder="Select destination" icon={MapIcon} />
          </div>

          <div className="input-group" style={{ marginTop: '30px' }}>
             <label style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Waves size={16} className="text-accent" /> Flood Severity Factor</span>
                <span className="badge-warning" style={{ padding: '4px 8px', borderRadius: '8px', fontSize: '1rem' }}>{floodFactor}x</span>
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
             <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', fontSize: '0.75rem', marginTop: '8px' }}>
                <span>Clear Traffic (0x)</span>
                <span>Severe Gridlock (4x)</span>
             </div>
          </div>

          <button className="btn btn-primary w-full mt-4" onClick={runDemo} disabled={loading || !source || !destination}>
            {loading ? "Simulating..." : "Run Flood Shock Demo"}
          </button>

          {error && <div className="error" style={{ marginTop: '16px', color: '#ef4444' }}>{error}</div>}
        </motion.section>

        <motion.section 
          className="glass-panel map-section"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          style={{ display: 'flex', flexDirection: 'column' }}
        >
          {demoResult ? (
            <>
              <div className="demo-kpis" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '20px', marginTop: 0 }}>
                <div className="kpi" style={{ borderLeft: '4px solid #7f1d1d', alignItems: 'flex-start' }}>
                  <div style={{ width: '100%' }}>
                    <span>Flooded A*</span>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginTop: '4px' }}>
                      <strong style={{ fontSize: '1.5rem' }}>{demoResult.flooded_astar_time_min} min</strong>
                      <strong style={{ fontSize: '1.1rem', color: '#7f1d1d' }}>{demoResult.flooded_distance_km} km</strong>
                    </div>
                  </div>
                </div>
                <div className="kpi" style={{ borderLeft: '4px solid #1e3a8a', alignItems: 'flex-start' }}>
                  <div style={{ width: '100%' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><BrainCircuit size={12} /> SAC Agent</span>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginTop: '4px' }}>
                      <strong style={{ fontSize: '1.5rem' }}>{demoResult.sac_time_min} min</strong>
                      <strong style={{ fontSize: '1.1rem', color: '#1e3a8a' }}>{demoResult.sac_distance_km} km</strong>
                    </div>
                  </div>
                </div>
                <div className="kpi" style={{ background: 'rgba(139, 92, 246, 0.1)', borderColor: 'var(--accent-primary)', alignItems: 'center' }}>
                  <div>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><Zap size={12} className="text-accent" /> SAC Time Saved</span>
                    <strong className="gradient-text" style={{ fontSize: '1.75rem' }}>{demoResult.sac_gain_vs_flooded_pct}%</strong>
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', marginBottom: '12px', padding: '0 8px' }}>
                <div className="demo-distances" style={{ margin: 0, display: 'flex', gap: '16px' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><span style={{ width: '12px', height: '3px', background: '#7f1d1d', display: 'inline-block' }}></span> Flooded Route</span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><span style={{ width: '12px', height: '3px', background: '#1e3a8a', display: 'inline-block' }}></span> SAC Route</span>
                </div>
              </div>

              <FloodDemoMap
                floodedPath={demoResult.flooded_astar_path || []}
                sacPath={demoResult.sac_path || []}
                sourceMarker={demoResult.source_marker || null}
                destinationMarker={demoResult.destination_marker || null}
                floodedExplored={demoResult.flooded_explored_nodes || []}
                sacExplored={demoResult.sac_explored_nodes || []}
              />
            </>
          ) : (
             <div className="mapEmpty" style={{ flex: 1, minHeight: '500px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', gap: '20px' }}>
                <Activity size={48} className="muted-icon" style={{ color: 'var(--accent-primary)' }} />
                <h3 style={{ fontSize: '1.5rem', margin: 0 }}>Ready to simulate</h3>
                <p className="muted" style={{ maxWidth: '400px', textAlign: 'center' }}>
                  Choose your route and set the severity of the bottleneck to see how the SAC algorithm adapts its path.
                </p>
             </div>
          )}
        </motion.section>
      </div>
    </motion.div>
  );
}
