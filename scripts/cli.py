
import argparse
import csv
import json
import os
import sys
import yaml

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import verify_pair

def _print_result(r):
    print(f"Image A: {r['image_a']}")
    print(f"  Image B: {r['image_b']}")
    print(f"Score : {r['score']:.6f}")
    print(f"Threshold: {r['threshold']:.6f}")
    print(f"Decision: {r['decision']}")
    print(f"Confidence: {r['confidence']:.4f}")
    print(f" Latency: {r['latency']['total_ms']:.2f} ms")


def _run_batch(csv_path,threshold,max_pairs=None):
    results = []
    with open(csv_path,newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_pairs and i >= max_pairs:
                break
            r = verify_pair(row["left_path"],row["right_path"], threshold)
            results.append(r)
    return results


def _print_batch_summary(results):
    import numpy as np
    latencies = [r["latency"]["total_ms"] for r in results]
    n_same = sum(1 for r in results if r["decision"]=="same")
    print(f"BATCH SUMMARY")
    print(f"Total pairs: {len(results)}")
    print(f"  Same decisions: {n_same}")
    print(f"Diff decisions: {len(results) - n_same}")
    print(f"Avg latency:{np.mean(latencies):.2f} ms")
    print(f"p50 latency:{np.percentile(latencies, 50):.2f} ms")
    print(f"p95 latency: {np.percentile(latencies, 95):.2f} ms")



def main():
    parser=argparse.ArgumentParser(description="Face Verification CLI — Milestone 3")
    parser.add_argument("--config",   default="configs/m3.yaml",help="Path to YAML config file")
    parser.add_argument("--image-a",help="Path to first image (single-pair mode)")
    parser.add_argument("--image-b",help="Path to second image (single-pair mode)")
    parser.add_argument("--pairs-csv",help="CSV file with left_path,right_path columns (batch mode)")
    parser.add_argument("--threshold",type=float, default=None,help="Override threshold from config")
    parser.add_argument("--max-pairs",type=int, default=None,help="Max pairs to process in batch mode")
    parser.add_argument("--output-json",default=None,help="Save results to JSON file")
    args = parser.parse_args()


    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    threshold = args.threshold if args.threshold is not None \
                else cfg["inference"]["threshold"]

    print(f"Face Verification CLI|threshold= {threshold:.6f}")

    if args.image_a and args.image_b:
        result = verify_pair(args.image_a, args.image_b, threshold)
        _print_result(result)
        all_results = [result]

    elif args.pairs_csv:
        all_results = _run_batch(args.pairs_csv, threshold, args.max_pairs)
        for r in all_results:
            _print_result(r)
            print("-" * 60)
        _print_batch_summary(all_results)

    else:
        parser.error("Provide either --image-a + --image-b  OR  --pairs-csv")
        return

    # Optional JSON output
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

if __name__ == "__main__":
    main()
    