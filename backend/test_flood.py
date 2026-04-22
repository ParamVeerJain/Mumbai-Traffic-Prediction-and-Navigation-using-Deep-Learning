import sys
import traceback
sys.path.append("e:/Projects/Major 2.0/Mumbai-Traffic-Prediction-and-Navigation-using-Deep-Learning/backend/data generation")
from smart_navigation_api import route_flood_demo, RouteRequest

try:
    req = RouteRequest(
        source='Goregaon East', 
        destination='Goregaon West', 
        start_datetime='2024-07-01T08:00', 
        algorithm='astar'
    )
    print("Calling route_flood_demo...", flush=True)
    res = route_flood_demo(req)
    print("Success!", flush=True)
except Exception as e:
    traceback.print_exc()
