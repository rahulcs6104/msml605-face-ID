import argparse
import json
import os
import sys
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import extract_embedding
from src.similarity import cosine_similarity_vectorized
from src.pairs import load_pairs
from src.metrics import threshold_sweep, select_threshold
from src.validation import validate_pairs


def main():
    parser=argparse.ArgumentParser(description="Re-calibrate threshold for embedding-based system")
    parser.add_argument("--config", default="configs/m3.yaml")
    parser.add_argument("--pairs-csv", default=None,help="Override val pairs path")
    parser.add_argument("--output", default="configs/m3_threshold.json",help="Where to save the selected threshold")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ec = cfg["evaluation"]
    pairs_csv=args.pairs_csv or os.path.join(cfg["outputs"]["pairs_dir"], "val_pairs.csv")

    print(f"loading the val pairs from: {pairs_csv}")
    pairs=load_pairs(pairs_csv)
    validate_pairs(pairs, check_paths=True)
    print(f"loaded {len(pairs)} val pairs")
    print("extracting embeddings (this may take a few minutes) ...")
    n=len(pairs)
    scores = np.zeros(n, dtype=np.float64)
    labels = np.zeros(n, dtype=np.int32)
    for i, pair in enumerate(pairs):
        emb_a = extract_embedding(pair["left_path"])
        emb_b = extract_embedding(pair["right_path"])
        scores[i] = float(cosine_similarity_vectorized(emb_a.reshape(1, -1), emb_b.reshape(1, -1))[0])
        labels[i] = int(pair["label"])
        if ((i+1)%50) == 0:
            print(f"processed {i + 1}/{n} pairs")

    thresholds = np.linspace(ec["threshold_min"], ec["threshold_max"], int(ec["threshold_steps"]))
    rule=ec.get("threshold_rule", "max_balanced_accuracy")
    sweep=threshold_sweep(scores, labels, thresholds)
    selected=select_threshold(sweep, rule=rule)

    print(f"\n###### Selected threshold ({rule}): {selected:.6f} #########")
    print("####### COPY THIS VALUE and update configs/m3.yaml → inference.threshold #######\n")


    result = {"threshold": selected,"rule": rule,"n_pairs": n,"representation": "facenet_vggface2_512d","metric": "cosine","score_stats": {"mean": round(float(np.mean(scores)), 6),"std":  round(float(np.std(scores)), 6),"min":  round(float(np.min(scores)), 6),"max":  round(float(np.max(scores)), 6),},}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved the threshold info to ====>> {args.output}")


if __name__ == "__main__":
    main()