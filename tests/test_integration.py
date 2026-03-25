import json
import os
import numpy as np
import pytest
from PIL  import Image

from src.evaluation import score_pairs, apply_threshold
from src.metrics import compute_metrics, threshold_sweep, select_threshold
from src.tracking import log_run
from src.validation  import validate_pairs, validate_scores

def write_random_images_and_build_pairs(tmp_dir, n=20):
    rng = np.random.default_rng(99)
    pairs = []
    for i in range(n):
        for side in ("left", "right"):
            pixels = rng.integers(0, 255, (10, 10, 3), dtype=np.uint8)
            img_path =os.path.join(tmp_dir, f"img_{i}_{side}.jpg")
            Image.fromarray(pixels).save(img_path)
        pairs.append({
            "left_path":  os.path.join(tmp_dir, f"img_{i}_left.jpg"),"right_path":os.path.join(tmp_dir, f"img_{i}_right.jpg"),
            "label":"1" if i < n // 2 else "0","split":"val"
        })
    return pairs


class TestIntegration:

    def test_full_pipeline_runs_cleanly(self,tmp_path):
        pairs =write_random_images_and_build_pairs(str(tmp_path), n=20)
        assert validate_pairs(pairs  , check_paths=True)
        scores,labels =score_pairs(pairs,metric="cosine",image_size=(10, 10))
        assert scores.shape==(20,)
        assert labels.shape ==(20,)
        assert validate_scores(scores, pairs)
        candidate_thresholds =np.linspace(-1.0, 1.0, 50)
        sweep_results =threshold_sweep(scores,labels, candidate_thresholds)
        assert len(sweep_results) ==50

        best_threshold = select_threshold(sweep_results, rule="max_balanced_accuracy")
        assert -1.0<= best_threshold  <= 1.0

        preds =apply_threshold(scores, best_threshold)
        metrics= compute_metrics(labels,preds)

        assert 0.0 <=metrics["accuracy"]<= 1.0
        for i in ("balanced_accuracy","f1",  "confusion_matrix"):
            assert i in metrics,f"missing key:{i}"

        runs_dir =str(tmp_path/"runs")
        log_run(runs_dir=runs_dir,run_id="integ_test",config_name="configs/m2.yaml",split="val",
            data_version="synthetic",threshold=best_threshold,metrics=metrics,note="integration test"
        )

        run_file = os.path.join(runs_dir,"integ_test.json")
        assert os.path.exists(run_file),"individual run file was not created"
        summary_path = os.path.join(runs_dir,"run_summary.json")
        summary = json.load(open(summary_path))
        assert len(summary)== 1
        assert summary[0]["run_id"] =="integ_test"
        assert   "accuracy"in summary[0]["metrics"]
        assert "confusion_matrix" in summary[0]["metrics"]