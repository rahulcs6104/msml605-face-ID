import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _verify_one(args_tuple):
    image_a, image_b, threshold = args_tuple
    from src.inference import verify_pair
    return verify_pair(image_a, image_b, threshold)


def main():
    parser = argparse.ArgumentParser(description="Load test for face verification system")
    parser.add_argument("--config", default="configs/m3.yaml")
    parser.add_argument("--pairs-csv", default=None,help="CSV with left_path,right_path columns")
    parser.add_argument("--max-pairs", type=int, default=20,help="Number of pairs to process (default 20)")
    parser.add_argument("--workers", type=int, default=4,help="Number of parallel workers (default 4)")
    parser.add_argument("--output", default="outputs/load_test_results.json",help="Where to save the JSON summary")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    threshold = cfg["inference"]["threshold"]
    pairs_csv = args.pairs_csv or cfg.get("load_test", {}).get("pairs_csv", os.path.join(cfg["outputs"]["pairs_dir"], "test_pairs.csv"))

    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        pairs = []
        for i, row in enumerate(reader):
            if i >= args.max_pairs:
                break
            pairs.append((row["left_path"], row["right_path"], threshold))

    print(f"load test configuration")
    print("#########################################################################")
    print(f"Pairs: {len(pairs)}")
    print(f"Workers: {args.workers}")
    print(f"Threshold: {threshold}")
    print(f"Source CSV : {pairs_csv}")
    print("==================================================\n")

    results = []
    failures = 0
    wall_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_verify_one, p): i for i, p in enumerate(pairs)}
        for i in as_completed(futures):
            idx = futures[i]
            try:
                result = i.result()
                results.append(result)
                print(f"  [{len(results):3d}/{len(pairs)}] "
                      f"latency={result['latency']['total_ms']:.1f}ms "
                      f"decision={result['decision']}")
            except Exception as e:
                failures += 1
                print(f"FAILED pair {idx}: {e}")

    wall_elapsed = time.perf_counter() - wall_start
    latencies = np.array([r["latency"]["total_ms"] for r in results])

    summary = {"total_requests":len(pairs),"successful":len(results),"failures":failures,"workers":args.workers,"wall_clock_seconds":round(wall_elapsed, 3),"throughput_pairs_per_sec": round(len(results)/max(wall_elapsed, 1e-6), 2),
        "latency_ms": {
            "mean":   round(float(np.mean(latencies)), 2),
            "median": round(float(np.median(latencies)), 2),
            "p95":    round(float(np.percentile(latencies, 95)), 2),
            "p99":    round(float(np.percentile(latencies, 99)), 2),
            "min":    round(float(np.min(latencies)), 2),
            "max":    round(float(np.max(latencies)), 2),
        },
    }

    print("\n==================================================")
    print(f"LOAD TEST RESULTS")
    print("\n==================================================")
    print(f"Requests:{summary['total_requests']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failures: {summary['failures']}")
    print(f"Workers: {summary['workers']}")
    print(f"Wall clock: {summary['wall_clock_seconds']:.3f} s")
    print(f"Throughput: {summary['throughput_pairs_per_sec']:.2f} pairs/sec")
    print(f"Latency mean: {summary['latency_ms']['mean']:.2f} ms")
    print(f"Latency p50: {summary['latency_ms']['median']:.2f} ms")
    print(f"Latency p95 : {summary['latency_ms']['p95']:.2f} ms")
    print(f"Latency p99 : {summary['latency_ms']['p99']:.2f} ms")
    print(f"Latency max : {summary['latency_ms']['max']:.2f} ms")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()