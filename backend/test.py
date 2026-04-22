import traceback
import sys

print("Importing...", flush=True)
from smart_navigation_api import route, RouteRequest
print("Import complete.", flush=True)

try:
    req = RouteRequest(
        source='Goregaon East', 
        destination='Goregaon West', 
        start_datetime='2024-07-01T08:00', 
        algorithm='astar'
    )
    print("Calling route...", flush=True)
    res = route(req)
    print("Success!", flush=True)
except Exception as e:
    traceback.print_exc()
