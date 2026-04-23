"""
Concurrency benchmark for the /process/ endpoint.

Usage:
    python concurrent_test.py step.png "pc download button" -c 8 -n 32
    python concurrent_test.py step.png "pc download button" -c 16 -n 64 --warmup 2

Note:
    - Sends the same image+prompt for every request, so vLLM's prefix cache will
      hit aggressively. Real-world numbers (mixed prompts/images) will be slightly
      worse — this is a best-case throughput measurement.
"""
import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")


def one_request(image_bytes, image_name, instruction, timeout=120):
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{BASE_URL}/process/",
            data={"instruction": instruction},
            files={"image_file": (image_name, image_bytes, "image/png")},
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        if resp.status_code != 200:
            return elapsed, resp.status_code, None
        return elapsed, 200, resp.json()
    except Exception as e:
        return time.perf_counter() - start, -1, repr(e)


def percentile(sorted_xs, p):
    if not sorted_xs:
        return 0.0
    k = int(len(sorted_xs) * p / 100)
    return sorted_xs[min(k, len(sorted_xs) - 1)]


def fmt_table(rows):
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))


def run_burst(image_bytes, image_name, instruction, concurrency, total, label=""):
    print(f"\n=== {label}  concurrency={concurrency}  total={total} ===")
    t0 = time.perf_counter()
    latencies = []
    statuses = []
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(one_request, image_bytes, image_name, instruction)
            for _ in range(total)
        ]
        for fut in as_completed(futures):
            elapsed, status, result = fut.result()
            latencies.append(elapsed)
            statuses.append(status)
            results.append(result)
    wall = time.perf_counter() - t0

    n_ok = sum(1 for s in statuses if s == 200)
    n_busy = sum(1 for s in statuses if s == 503)
    n_err = sum(1 for s in statuses if s not in (200, 503))

    latencies.sort()
    print(f"Wall time:   {wall:.2f}s")
    print(f"Throughput:  {n_ok/wall:.2f} req/s   ({n_ok} ok / {n_busy} 503 / {n_err} err)")
    print(f"Latency (s): min={min(latencies):.3f}  p50={percentile(latencies, 50):.3f}  "
          f"p90={percentile(latencies, 90):.3f}  p95={percentile(latencies, 95):.3f}  "
          f"p99={percentile(latencies, 99):.3f}  max={max(latencies):.3f}  "
          f"mean={statistics.mean(latencies):.3f}")

    # Result consistency check (all same coord since same image+prompt)
    ok_results = [r for r in results if isinstance(r, dict) and "x" in r]
    if ok_results:
        xs = [r["x"] for r in ok_results]
        ys = [r["y"] for r in ok_results]
        x_uniq = len(set(round(x, 1) for x in xs))
        y_uniq = len(set(round(y, 1) for y in ys))
        consistent = "✓" if (x_uniq == 1 and y_uniq == 1) else f"⚠ x:{x_uniq} y:{y_uniq} unique"
        print(f"Result:      ({xs[0]:.1f}, {ys[0]:.1f})   consistency: {consistent}")

    return wall, n_ok / wall if wall > 0 else 0, latencies


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("instruction")
    p.add_argument("-c", "--concurrency", type=int, default=8)
    p.add_argument("-n", "--total", type=int, default=32)
    p.add_argument("-w", "--warmup", type=int, default=2,
                   help="warmup requests before measurement (default 2)")
    p.add_argument("--sweep", action="store_true",
                   help="run a sweep across concurrency=[1,2,4,8,16] with --total each")
    args = p.parse_args()

    with open(args.image, "rb") as f:
        image_bytes = f.read()
    image_name = os.path.basename(args.image)

    print(f"Server:      {BASE_URL}")
    print(f"Image:       {args.image}  ({len(image_bytes)/1024:.1f} KB)")
    print(f"Instruction: {args.instruction!r}")

    # Health check
    try:
        h = requests.get(BASE_URL + "/", timeout=5)
        print(f"Health:      {h.status_code}  {h.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        sys.exit(1)

    # Warmup (sequential, single request each)
    if args.warmup > 0:
        print(f"\nWarmup ({args.warmup} sequential requests)...")
        for i in range(args.warmup):
            t = time.perf_counter()
            one_request(image_bytes, image_name, args.instruction)
            print(f"  warmup {i+1}: {(time.perf_counter()-t)*1000:.0f} ms")

    if args.sweep:
        print("\n=== SWEEP ===")
        rows = [["concurrency", "wall(s)", "qps", "p50(s)", "p95(s)", "p99(s)"]]
        for c in [1, 2, 4, 8, 16]:
            wall, qps, lat = run_burst(image_bytes, image_name, args.instruction, c, args.total, label=f"c={c}")
            rows.append([c, f"{wall:.2f}", f"{qps:.2f}",
                         f"{percentile(lat, 50):.3f}",
                         f"{percentile(lat, 95):.3f}",
                         f"{percentile(lat, 99):.3f}"])
        print("\n=== SWEEP SUMMARY ===")
        fmt_table(rows)
    else:
        run_burst(image_bytes, image_name, args.instruction, args.concurrency, args.total)


if __name__ == "__main__":
    main()
