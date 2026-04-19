import argparse
import os
import subprocess
import sys
from typing import List


ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(ROOT, "backend", "data generation", "data")


def _pred_path(h: int) -> str:
    return os.path.join(DATA_ROOT, "predictions", f"horizon_tplus_{h:02d}.parquet")


def _frames_path(region: str, h: int) -> str:
    return os.path.join(DATA_ROOT, "pngs", f"tplus_{h:02d}", region, "frames.json")


def _run(cmd: List[str]) -> None:
    print(">", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _ensure_predictions(horizons: List[int]) -> None:
    missing = [h for h in horizons if not os.path.exists(_pred_path(h))]
    if not missing:
        print("Predictions already available for all requested horizons.")
        return
    print(f"Generating predictions for missing horizons: {missing}")
    _run(
        [
            sys.executable,
            os.path.join("backend", "data generation", "generate_predictions.py"),
            "--min_h",
            str(min(missing)),
            "--max_h",
            str(max(missing)),
        ]
    )


def _ensure_pngs(horizons: List[int], force: bool) -> None:
    for h in horizons:
        all_regions_ready = all(
            os.path.exists(_frames_path(region, h)) for region in ("mumbai", "goregaon")
        )
        if all_regions_ready and not force:
            print(f"PNG frames already available for t+{h}.")
            continue
        print(f"Generating PNGs for t+{h} ...")
        _run(
            [
                sys.executable,
                os.path.join("backend", "data generation", "generate_pngs.py"),
                "--horizon",
                str(h),
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate missing assets and launch frontend.")
    parser.add_argument("--min_h", type=int, default=1, help="Start horizon (1..12)")
    parser.add_argument("--max_h", type=int, default=12, help="End horizon (1..12)")
    parser.add_argument(
        "--force_png",
        action="store_true",
        help="Regenerate PNG metadata/frames even if already present.",
    )
    parser.add_argument(
        "--no_launch",
        action="store_true",
        help="Only prepare data; do not launch frontend.",
    )
    args = parser.parse_args()

    min_h = max(1, int(args.min_h))
    max_h = min(12, int(args.max_h))
    if min_h > max_h:
        raise ValueError("--min_h must be <= --max_h")

    horizons = list(range(min_h, max_h + 1))
    print(f"Preparing horizons: {horizons}")
    _ensure_predictions(horizons)
    _ensure_pngs(horizons, force=bool(args.force_png))

    if args.no_launch:
        print("Preparation done. Frontend launch skipped by --no_launch.")
        return

    print("Launching frontend app ...")
    _run([sys.executable, os.path.join("frontend", "app.py")])


if __name__ == "__main__":
    main()
