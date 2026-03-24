# run.py
import os, sys

def main():
    from config import DATA_DIR

    graph_ok = os.path.exists(os.path.join(DATA_DIR, "graph.pkl"))
    idx_ok   = os.path.exists(os.path.join(DATA_DIR, "road_to_idx.pkl"))

    if not graph_ok or not idx_ok:
        print("=" * 60)
        print("First run detected. Generating dummy data...")
        print("=" * 60)
        import subprocess
        subprocess.run([sys.executable, "scripts/generate_dummy.py"], check=True)

    print("\nStarting UI at http://localhost:8050\n")
    from ui.app import app
    app.run(debug=False, host="0.0.0.0", port=8050)   # ✅ bind all interfaces


if __name__ == "__main__":
    main()