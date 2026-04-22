import sys
sys.path.append("e:/Projects/Major 2.0/Mumbai-Traffic-Prediction-and-Navigation-using-Deep-Learning/backend/data generation")
import tda_router_goregaon as tda

print("Loading graph...", flush=True)
g, ctx, _ = tda.load_goregaon_graph_and_context(None)
print("Graph loaded.", flush=True)

src_name = "Goregaon East"
dst_name = "Goregaon West"

location_lookup = {name: (float(lat), float(lon)) for name, lat, lon in tda.NAMED_LOCATIONS}
src_lat, src_lon = location_lookup[src_name]
dst_lat, dst_lon = location_lookup[dst_name]

src = tda.ox.distance.nearest_nodes(g, src_lon, src_lat)
dst = tda.ox.distance.nearest_nodes(g, dst_lon, dst_lat)

print(f"Calling tda_star src={src} dst={dst}", flush=True)
path, total_sec, explored = tda.tda_star(g, ctx, src, dst, start_hour=8, start_elapsed_sec=0.0, algorithm="astar")
print(f"Found path of length {len(path) if path else 0}, explored nodes: {len(explored)}", flush=True)
