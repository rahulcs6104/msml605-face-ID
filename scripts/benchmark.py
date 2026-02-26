#This file is for benchmarking cosine's and euclidean's loop vs vectorized results
#and to see which one performs well
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.similarity import (cosine_similarity_loop,cosine_similarity_vectorized,euclidean_distance_loop,euclidean_distance_vectorized)


def run_benchmark(N: int, D: int, seed: int):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((N, D)).astype(np.float64)
    b = rng.standard_normal((N, D)).astype(np.float64)

    print(f"\n vectors ==> N=>{N}, D=>{D}, dtype=>{a.dtype}")
    print("======================================================================================================================================")

    results = {"N": N, "D": D, "seed": seed}

    print("\n COSINE SIMILARITY : ")
    start_time = time.perf_counter()
    cosine_loop = cosine_similarity_loop(a, b)
    time_cosine_loop = time.perf_counter() - start_time

    start_time = time.perf_counter()
    cosine_vec = cosine_similarity_vectorized(a, b)
    time_cosine_vec = time.perf_counter() - start_time

    cosine_max_diff = float(np.max(np.abs(cosine_loop - cosine_vec)))
    cos_tolerance = 1e-10
    cosine_correct = cosine_max_diff < cos_tolerance

    if time_cosine_vec > 0:
        speed_c = time_cosine_loop / time_cosine_vec
    else:
        speed_c= float("inf")

    print(f"  Loop time        : {time_cosine_loop:.4f}s")
    print(f"  Vectorized time  : {time_cosine_vec:.4f}s")
    print(f"  Speed          : {speed_c:.1f}x")
    print(f"  Max abs diff     : {cosine_max_diff:.2e}  (tolerance={cos_tolerance})")
    print(f"  Correctness check: {'PASSED' if cosine_correct else 'FAILED'}")

    results["cosine"] = {
        "loop_time_s": round(time_cosine_loop, 6),
        "vec_time_s": round(time_cosine_vec, 6),
        "speedup_x": round(speed_c, 2),
        "max_abs_diff": cosine_max_diff,
        "tolerance": cos_tolerance,
        "correctness_pass": cosine_correct,
    }

    print("\n EUCLIDEAN DISTANCE")

    start_time = time.perf_counter()
    euclidean_loop = euclidean_distance_loop(a, b)
    time_euclide_loop = time.perf_counter() - start_time

    start_time = time.perf_counter()
    euclidean_vec = euclidean_distance_vectorized(a, b)
    time_euclidean_vec = time.perf_counter() - start_time

    euc_max_diff = float(np.max(np.abs(euclidean_loop - euclidean_vec)))
    euc_tolerance = 1e-10
    euc_correct = euc_max_diff < euc_tolerance

    if time_euclidean_vec > 0:
        speed_e = time_euclide_loop / time_euclidean_vec
    else:
        speed_e = float("inf")

    print(f"  Loop time        : {time_euclide_loop:.4f}s")
    print(f"  Vectorized time  : {time_euclidean_vec:.4f}s")
    print(f"  Speed            : {speed_e:.1f}x")
    print(f"  Max abs diff     : {euc_max_diff:.2e}  (tolerance={euc_tolerance})")
    print(f"  Correctness check: {'PASSED' if euc_correct else 'FAILED'}")

    results["euclidean"] = {"loop_time_s": round(time_euclide_loop, 6),"vec_time_s": round(time_euclidean_vec, 6),"speedup_x": round(speed_e, 2),"max_abs_diff": euc_max_diff,"tolerance": euc_tolerance,"correctness_pass": euc_correct}

    print("\n")
    print("======================================================================================================================================")
    return results

def main():
    parser = argparse.ArgumentParser(description="benchmarking looped vs vectorized cosine and euclidean")
    parser.add_argument("--config", required=True, help="Path to the YAML config file(m1.yaml)")
    args = parser.parse_args()

    with open(args.config) as f:
        config_values = yaml.safe_load(f)

    N = config_values["benchmark"]["N"]
    D =config_values["benchmark"]["D"]
    seed=config_values["seed"]

    results = run_benchmark(N=N,D=D,seed=seed)
    bench_dir = Path(config_values["outputs"]["bench_dir"])
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_path = bench_dir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f,indent=2)
    print("\n")
    print(f"Results saved to this:  {out_path}")


if __name__ == "__main__":
    main()