"""Measure the real service: start uvicorn, fire concurrent requests, report latency percentiles.

    python scripts/loadtest.py --requests 3000 --concurrency 16
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
BODY = {"pickup_zone": 161, "dropoff_zone": 237, "pickup_time": "2023-10-14T18:30:00", "distance_miles": 2.4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "tripeta.api:app", "--port", str(a.port),
                               "--workers", str(a.workers), "--log-level", "warning"], cwd=ROOT,
                              env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    base = f"http://127.0.0.1:{a.port}"
    try:
        for _ in range(60):
            try:
                if httpx.get(base + "/ready", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        lat = []

        def one(_):
            with httpx.Client(base_url=base, timeout=10) as c:
                out = []
                for _ in range(a.requests // a.concurrency):
                    t = time.perf_counter()
                    r = c.post("/predict", json=BODY)
                    r.raise_for_status()
                    out.append((time.perf_counter() - t) * 1000)
                return out

        t0 = time.perf_counter()
        with ThreadPoolExecutor(a.concurrency) as ex:
            for part in ex.map(one, range(a.concurrency)):
                lat += part
        wall = time.perf_counter() - t0
        lat.sort()
        q = lambda p: lat[min(int(len(lat) * p), len(lat) - 1)]  # noqa: E731
        print(json.dumps({"requests": len(lat), "concurrency": a.concurrency, "workers": a.workers,
                          "throughput_rps": round(len(lat) / wall, 1), "p50_ms": round(q(.5), 1),
                          "p95_ms": round(q(.95), 1), "p99_ms": round(q(.99), 1),
                          "mean_ms": round(statistics.mean(lat), 1)}, indent=2))
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
