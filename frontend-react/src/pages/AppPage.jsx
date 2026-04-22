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
import CustomSelect from "../components/CustomSelect";
import { Gauge, MapPinned, Navigation, Route, Timer, Activity, Map as MapIcon, Compass, BrainCircuit } from "lucide-react";
import { motion } from "framer-motion";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function FitMapToRoute({ bounds }) {
  const map = useMap();

  useEffect(() => {
    // Invalidate size in case the container was hidden or resized
    setTimeout(() => {
      map.invalidateSize();
      if (bounds && bounds.length > 0) {
        map.fitBounds(bounds, { padding: [32, 32] });
      }
    }, 100);
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
    return segmentLines.flat();
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
            <Polyline key={`route-main-${i}`} positions={positions} pathOptions={{ color: "#000000", weight: 6, opacity: 1 }} />
          ))}

          {path.map((seg, i) => (
            <Polyline
              key={`${seg.road_name}-${seg.edge_key}-${i}`}
              positions={segmentLines[i]}
              pathOptions={{ color: "#000000", weight: 12, opacity: 0 }}
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
      {!path.length && <div className="mapEmpty"><MapPinned size={48} className="muted-icon"/><span>Select route and click Find Best Route</span></div>}
    </div>
  );
}



export default function AppPage() {
  const [locations, setLocations] = useState([]);
  const [source, setSource] = useState("");
  const [destination, setDestination] = useState("");
  const [startDatetime, setStartDatetime] = useState("2024-07-01T08:00");
  const [algorithm, setAlgorithm] = useState("astar");
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
        start_datetime: startDatetime,
        algorithm
      });
      setResult(res.data);
    } catch (e) {
      setResult(null);
      setError(e?.response?.data?.detail || "Unable to find route right now.");
    } finally {
      setLoading(false);
    }
  };



  const locationOptions = useMemo(() => {
    return locations.map(loc => ({ label: loc, value: loc }));
  }, [locations]);

  return (
    <motion.div 
      className="app-page"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <header className="app-header">
        <h1>Live Traffic <span className="gradient-text">Intelligence</span></h1>
        <p>Real-time routing for Goregaon with predicted traffic delays.</p>
      </header>

      <div className="app-grid">
        <motion.section 
          className="glass-panel controls"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <h2>Trip Planner</h2>
          
          <div className="input-group">
             <label>Source</label>
             <CustomSelect 
               value={source} 
               onChange={setSource} 
               options={locationOptions} 
               placeholder="Select source"
               icon={Compass}
             />
          </div>

          <div className="input-group">
             <label>Destination</label>
             <CustomSelect 
               value={destination} 
               onChange={setDestination} 
               options={locationOptions} 
               placeholder="Select destination"
               icon={MapIcon}
             />
          </div>

          <div className="input-group">
             <label>Start Date & Time</label>
             <input type="datetime-local" value={startDatetime} onChange={(e) => setStartDatetime(e.target.value)} />
          </div>


          <button className="btn btn-primary w-full mt-4" onClick={runRoute} disabled={loading || !source || !destination}>
            {loading ? "Analyzing Traffic..." : "Find Best Route"}
          </button>

          {error && <div className="error">{error}</div>}
        </motion.section>

        <motion.section 
          className="glass-panel map-section"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <div className="kpis">
            <div className="kpi">
              <Timer size={18} className="text-accent" />
              <div>
                 <span>Est. Time</span>
                 <strong>{result ? `${result.est_time_min} min` : "--"}</strong>
              </div>
            </div>
            <div className="kpi">
              <Navigation size={18} className="text-accent" />
              <div>
                 <span>Est. Distance</span>
                 <strong>{result ? `${result.est_distance_km} km` : "--"}</strong>
              </div>
            </div>
            <div className="kpi">
              <Route size={18} className="text-accent" />
              <div>
                 <span>Segments</span>
                 <strong>{result ? result.segments_used : "--"}</strong>
              </div>
            </div>
          </div>
          <MapView
            path={result?.path || []}
            sourceMarker={result?.source_marker || null}
            destinationMarker={result?.destination_marker || null}
            exploredNodes={result?.explored_nodes || []}
          />
        </motion.section>
      </div>

      <motion.section 
        className="glass-panel delayCard"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <div className="delay-header">
           <h2>
             <Gauge size={22} className="text-accent" /> Critical Bottlenecks
           </h2>
           <p className="subtitle">Roads causing the most delay on your route.</p>
        </div>
        
        {!result?.delays?.length ? (
          <div className="empty-state">
             <Activity size={32} />
             <p className="muted">No route calculated yet. Run Smart Navigation to see delay insights.</p>
          </div>
        ) : (
          <div className="delayList">
            {result.delays.map((d, idx) => (
              <motion.article 
                 key={`${d.road_name}-${idx}`} 
                 className="delayItem glass-subpanel"
                 initial={{ opacity: 0, y: 10 }}
                 animate={{ opacity: 1, y: 0 }}
                 transition={{ delay: 0.5 + idx * 0.1 }}
              >
                <h3>{d.road_name}</h3>
                <div className="delay-meta">
                   <p>
                     <MapPinned size={14} className="text-accent" /> Near {d.near_area}
                   </p>
                   <div className="delay-stats">
                      <span className="badge-danger">{d.delay_seconds.toFixed(0)}s delay</span>
                      <span className="badge-warning">{d.traffic_factor.toFixed(2)}x traffic</span>
                   </div>
                   <p className="text-sm muted mt-2">Segment length: {d.segment_length_m.toFixed(0)}m</p>
                </div>
              </motion.article>
            ))}
          </div>
        )}
      </motion.section>

    </motion.div>
  );
}
