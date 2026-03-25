
import argparse, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import score_pairs, apply_threshold
from src.metrics import compute_metrics, threshold_sweep, select_threshold, roc_data
from src.pairs import load_pairs
from src.tracking import log_run
from src.validation import validate_pairs, validate_scores, validate_config


def _roc_plot(fprs, tprs, run_id, split, metric, path):
    sorted_idx = np.argsort(fprs)
    fprs_sorted = fprs[sorted_idx]
    tprs_sorted = tprs[sorted_idx]
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fprs_sorted,tprs_sorted,linewidth=2,label=f"{metric} pixel similarity")

    ax.plot([0, 1],[0, 1],linestyle="--",alpha=0.4,label="Random")
    ax.set_xlabel("false positive")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC curve ==> {split} ({run_id})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"roc saved to this path => {path}")



def _cm_plot(cm, threshold, split, run_id, path):
    matrix = np.array([
        [cm["TN"], cm["FP"]],
        [cm["FN"], cm["TP"]]
    ])
    figure, ax = plt.subplots(figsize=(4, 4))
    a = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["predicted : different", "predicted : same"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["actual : different", "actual : same"])
    max_val = matrix.max()
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            if value>max_val/1.5:
                text_color="white"
            else:
                text_color="black"
            ax.text(j,i,str(value),ha="center",va="center",fontsize=13,fontweight="bold",color=text_color)

    ax.set_title(f"Confusion Matrix => {split} (run {run_id}, t={threshold:.4f})")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved confusion matrix to the path => {path}")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--split",required=True, choices=["train", "val", "test"])
    p.add_argument("--run-id",required=True)

    p.add_argument("--threshold",type=float, default=None)
    p.add_argument("--pairs-dir",default=None)
    p.add_argument("--data-version",default="baseline")
    p.add_argument("--note",default="")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    ec= cfg["evaluation"]
    runs_dir= cfg["tracking"]["runs_dir"]
    pairs_dir= args.pairs_dir or cfg["outputs"]["pairs_dir"]
    metric=ec.get("metric", "cosine")
    img_size  =tuple(ec.get("image_size", [50, 50]))
    pairs_path = os.path.join(pairs_dir, f"{args.split}_pairs.csv")
    if not os.path.exists(pairs_path):
        sys.exit(f"error ==> {pairs_path} is not found.Please run the pair creation code first.")

    pairs = load_pairs(pairs_path)
    print(f"loaded {len(pairs)} pairs [{args.split}]")
    validate_pairs(pairs,check_paths=True)

    print(f"Scoring with metric='{metric}',image_size={img_size} ...")
    scores, labels = score_pairs(pairs, metric=metric, image_size=img_size)
    validate_scores(scores, pairs)
    thresholds = np.linspace(ec["threshold_min"], ec["threshold_max"],int(ec["threshold_steps"]))
    os.makedirs(runs_dir, exist_ok=True)

    if args.threshold is None:
        print("currnntly running threshold sweep ")
        sweep= threshold_sweep(scores, labels, thresholds)
        rule= ec.get("threshold_rule","max_balanced_accuracy")
        selected=select_threshold(sweep, rule=rule)
        print(f"\n*** selected threshold ({rule}) : {selected:.6f} ***")
        print(" ***************   COPY THIS VALUE ==>> you can use it with --threshold for the next run ***************\n")
        sweep_path = os.path.join(runs_dir, f"{args.run_id}_sweep.json")
        with open(sweep_path, "w") as f:
            json.dump({"rule": rule, "selected": selected,
                       "n_thresholds": len(thresholds), "sweep": sweep}, f, indent=2)
        fprs,tprs= roc_data(scores, labels, thresholds)
        _roc_plot(fprs, tprs, args.run_id, args.split, metric,
                  os.path.join(runs_dir, f"{args.run_id}_roc.png"))
    else:
        selected = args.threshold
        print(f"Using fixed threshold: {selected}")

    predictions=apply_threshold(scores, selected)
    metrics=compute_metrics(labels, predictions)
    metrics["threshold_used"]=selected



    print(f"\n--- Results [{args.split}] threshold={selected:.6f} ---")
    for k, v in metrics.items():
        print(f"  {k:22s}: {v}")

    _cm_plot(metrics["confusion_matrix"], selected, args.split,args.run_id,
             os.path.join(runs_dir, f"{args.run_id}_cm.png"))

    log_run(runs_dir=runs_dir, run_id=args.run_id, config_name=args.config,
            split=args.split, data_version=args.data_version,
            threshold=selected, metrics=metrics, note=args.note)
    print(f"\nDone with run '{args.run_id}'")





if __name__ == "__main__":
    main()