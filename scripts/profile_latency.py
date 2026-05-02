#!/usr/bin/env python3
"""
Milestone 4 hardware-aware latency profiler.
Measures per-stage latency: preprocessing (MTCNN), embedding (FaceNet), scoring (cosine).
Also measures batch-size sensitivity (throughput vs batch size).
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.embeddings import preprocess_face, _get_model, _get_device
from src.pairs import load_pairs
from src.similarity import cosine_similarity_vectorized


def get_hardware_info():
    info = {"platform": platform.platform(),"python_version": platform.python_version(),"processor": platform.processor(),"cpu_count_logical": os.cpu_count()}
    try:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass


    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    return info


def _stats(arr):
    return {"mean_ms":round(float(np.mean(arr)), 3),"std_ms":round(float(np.std(arr)), 3),"p50_ms":round(float(np.percentile(arr, 50)), 3),"p95_ms":round(float(np.percentile(arr, 95)), 3),"min_ms":round(float(np.min(arr)), 3),"max_ms":round(float(np.max(arr)), 3),}


def _run_pair(model, device, pair):

    # Stage 1 => doing preprocessing {MTCNN face detection + resize} 
    t0 = time.perf_counter()
    face_a=preprocess_face(pair["left_path"])
    face_b=preprocess_face(pair["right_path"])
    pre_ms=(time.perf_counter()-t0)*1000

    # Stage 2: embedding
    t0 = time.perf_counter()
    with torch.no_grad():
        emb_a = model(face_a.unsqueeze(0).to(device)).cpu().numpy().flatten()
        emb_b = model(face_b.unsqueeze(0).to(device)).cpu().numpy().flatten()
    emb_ms = (time.perf_counter() - t0) * 1000

    # Stage 3: scoring , i ahve choosen cosine similarity
    t0 = time.perf_counter()
    a = float(cosine_similarity_vectorized(emb_a.reshape(1, -1), emb_b.reshape(1, -1))[0])
    sco_ms = (time.perf_counter() - t0)*1000

    return pre_ms, emb_ms, sco_ms


def profile_stages(pairs, model, device, n_pairs, warmup):
    print(f"warming up with {warmup} pairs")
    for p in pairs[:warmup]:
        _run_pair(model, device, p)

    pre_arr, emb_arr, sco_arr, tot_arr = [],[],[],[]
    actual = min(n_pairs, len(pairs))
    for i, p in enumerate(pairs[:actual]):
        pr, em, sc = _run_pair(model, device, p)
        pre_arr.append(pr)
        emb_arr.append(em)
        sco_arr.append(sc)
        tot_arr.append(pr + em + sc)
        if(i+1)%10==0:
            print(f"profiled {i + 1}/{actual} pairs")

    return {"n_pairs_measured": actual,"preprocessing_mtcnn":_stats(pre_arr),"embedding_facenet":_stats(emb_arr),"scoring_cosine":_stats(sco_arr),"end_to_end":_stats(tot_arr),}


def profile_batch_sensitivity(pairs, model, device, batch_sizes):
    results = []
    for bs in batch_sizes:
        actual = min(bs, len(pairs))
        t0 = time.perf_counter()
        for p in pairs[:actual]:
            _run_pair(model, device, p)
        elapsed_s = time.perf_counter() - t0
        row = {"batch_size":actual,"total_ms":round(elapsed_s * 1000, 2),"per_pair_ms":round(elapsed_s * 1000 / actual, 2),"throughput_pairs_per_sec":round(actual / (elapsed_s + 1e-9), 3),}
        results.append(row)
        print(f"  batch={actual:3d}: {row['total_ms']:8.1f} ms total | "
              f"{row['per_pair_ms']:7.1f} ms/pair | "
              f"{row['throughput_pairs_per_sec']:.2f} pairs/s")
    return results


def main():
    parser = argparse.ArgumentParser(description="M4 Latency Profiler")
    parser.add_argument("--config",default="configs/m3.yaml")
    parser.add_argument("--pairs-csv",default=None)
    parser.add_argument("--n-pairs",type=int, default=20)
    parser.add_argument("--warmup",type=int, default=3)
    parser.add_argument("--batch-sizes",default="1,5,10,20,50")
    parser.add_argument("--output",default="outputs/profiling_results.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pairs_csv = args.pairs_csv or os.path.join(
        cfg["outputs"]["pairs_dir"], "test_pairs.csv"
    )
    print(f"loading pairs from: {pairs_csv}")
    pairs = load_pairs(pairs_csv)
    print(f"loaded {len(pairs)} pairs")

    hw = get_hardware_info()
    print("\n--- Hardware ---")
    for k, v in hw.items():
        print(f"  {k}: {v}")

    print("\loading FaceNet model and MTCNN detector (one-time cost)")
    model  = _get_model()
    device = _get_device()

    print(f"\n--- Stage Latency (n={args.n_pairs}, warmup={args.warmup}) ---")
    stage = profile_stages(pairs, model, device,
                           n_pairs=args.n_pairs, warmup=args.warmup)
    print(f"\n  Preprocessing (MTCNN)  : {stage['preprocessing_mtcnn']['mean_ms']:7.1f} ms mean  "
          f"(p95={stage['preprocessing_mtcnn']['p95_ms']:.1f})")
    print(f"  Embedding   (FaceNet)  : {stage['embedding_facenet']['mean_ms']:7.1f} ms mean  "
          f"(p95={stage['embedding_facenet']['p95_ms']:.1f})")
    print(f"  Scoring     (cosine)   : {stage['scoring_cosine']['mean_ms']:7.3f} ms mean  "
          f"(p95={stage['scoring_cosine']['p95_ms']:.3f})")
    print(f"  End-to-end             : {stage['end_to_end']['mean_ms']:7.1f} ms mean  "
          f"(p95={stage['end_to_end']['p95_ms']:.1f})")

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    print(f"\n--- Batch-Size Sensitivity: {batch_sizes} ---")
    batch = profile_batch_sensitivity(pairs, model, device, batch_sizes)

    output = {"hardware":hw,"config":args.config,"stage_latency":stage,"batch_sensitivity":batch}

    os.makedirs(os.path.dirname(args.output) or ".",exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()