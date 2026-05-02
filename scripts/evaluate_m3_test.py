import argparse
import json
import os
import sys
import numpy as np
import yaml

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.embeddings import extract_embedding
from src.similarity import cosine_similarity_vectorized
from src.pairs import load_pairs
from src.metrics import compute_metrics
from src.evaluation import apply_threshold
from src.validation import validate_pairs

def main():
    parser=argparse.ArgumentParser(description="Evaluate M3 (FaceNet) on the test split and save metrics")
    parser.add_argument("--config",default="configs/m3.yaml")
    parser.add_argument("--pairs-csv",default=None, help="Override test pairs path")
    parser.add_argument("--output",default="outputs/m3_test_metrics.json")
    args =parser.parse_args()
    with open(args.config) as f:
        cfg =yaml.safe_load(f)
    threshold =cfg["inference"]["threshold"]
    pairs_csv =args.pairs_csv or os.path.join(cfg["outputs"]["pairs_dir"], "test_pairs.csv")
    print(f"Loading test pairs from: {pairs_csv}")
    pairs = load_pairs(pairs_csv)
    validate_pairs(pairs, check_paths=True)
    print(f"Loaded {len(pairs)} pairs | operating threshold = {threshold:.6f}")
    print("Extracting FaceNet embeddings (may take 10-20 min on CPU)")
    n=len(pairs)
    scores =np.zeros(n,dtype=np.float64)
    labels =np.zeros(n,dtype=np.int32)
    for i, pair in enumerate(pairs):
        emb_a =extract_embedding(pair["left_path"])
        emb_b =extract_embedding(pair["right_path"])
        scores[i] = float(cosine_similarity_vectorized(emb_a.reshape(1,-1),emb_b.reshape(1, -1))[0])
        labels[i] = int(pair["label"])
        if (i + 1)%100== 0:
            print(f"processed {i+ 1}/{n}")
    predictions =apply_threshold(scores,threshold)
    metrics =compute_metrics(labels,predictions)
    metrics["threshold_used"] =threshold
    metrics["n_pairs"] =n
    metrics["split"] ="test"
    metrics["model"] ="facenet_vggface2_512d"

    print(f"\n M3 Test Results(threshold= {threshold:.6f})")
    for k,v in metrics.items():
        print(f"{k:25s}:{v}")
    os.makedirs(os.path.dirname(args.output) or ".",exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics,f,indent=2)
    print(f"\nSaved to:{args.output}")
    print("Use these values in the SystemCard.")

if __name__ == "__main__":
    main()