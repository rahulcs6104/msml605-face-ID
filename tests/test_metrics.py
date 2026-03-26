"""tests for src/metrics.py"""
import numpy as np
import pytest
from src.metrics import (confusion_matrix, compute_metrics,threshold_sweep,select_threshold, roc_data)

class TestConfusionMatrix:

    def test_perfect_predictions(self):
        cm = confusion_matrix(np.array([1,1,0,0]), np.array([1,1,0,0]))

        assert cm == {"TP":2, "FP":0, "TN":2, "FN":0}

    def test_everything_flipped(self):
        # every prediction is wrong
        cm = confusion_matrix(np.array([1,1,0,0]), np.array([0,0,1,1]))

        assert cm["TP"] == 0 and cm["FP"] == 2

        assert cm["TN"] == 0 and cm["FN"] == 2

    def test_half_right(self):
        cm = confusion_matrix(np.array([1,0,1,0]), np.array([1,0,0,1]))

        assert cm["TP"] == 1 and cm["TN"] == 1

        assert cm["FP"] == 1 and cm["FN"] == 1

    def test_all_true_negatives(self):
        cm = confusion_matrix(np.array([0,0,0]), np.array([0,0,0]))

        assert cm["TN"] == 3

        assert cm["TP"] == cm["FP"] == cm["FN"] == 0



class TestComputeMetrics: #this is for the compute metrics

    def test_perfect_score(self):
        m =compute_metrics(np.array([1,1,0,0]), np.array([1,1,0,0]))

        assert m["accuracy"]==1.0
        assert m["balanced_accuracy"]==1.0
        assert m["f1"]==1.0 and m["fpr"]==0.0

    def test_worst_case(self):
        m =compute_metrics(np.array([1,1,0,0]), np.array([0,0,1,1]))
        assert m["accuracy"]==0.0 and m["balanced_accuracy"]==0.0

    def test_metrics_stay_in_01(self):
        rng=np.random.default_rng(42)
        y_true=rng.integers(0,2,100)
        y_pred=rng.integers(0,2,100)
        m=compute_metrics(y_true,y_pred)
        for i in ("accuracy", "balanced_accuracy", "f1", "tpr", "fpr", "precision"):
            assert 0.0 <= m[i] <= 1.0, f"{i} is out of [0,1]"

    def test_confusion_matrix_included(self):
        m=compute_metrics(np.array([1,0]),np.array([1,0]))
        assert "confusion_matrix" in m


class TestThresholdSweep:

    def test_output_length_matches_thresholds(self):
        thresholds=np.linspace(-1, 1, 20)

        res=threshold_sweep(np.array([0.1,0.5,0.8]),np.array([0,1,1]),thresholds)

        assert len(res)==20

    def test_each_entry_has_threshold_key(self):
        res=threshold_sweep(np.array([0.5]), np.array([1]), [0.3])

        assert "threshold" in res[0]

    def test_select_best_balanced_accuracy(self):
        scores=np.array([0.2, 0.2, 0.8, 0.8])

        labels=np.array([0,   0,   1,   1  ])
        sweep=threshold_sweep(scores, labels, np.array([0.3, 0.5, 0.7, 0.9]))
        best=select_threshold(sweep, "max_balanced_accuracy")
        assert best in [0.3, 0.5, 0.7]

    def test_select_best_f1(self):
        scores=np.array([0.1, 0.9])
        labels=np.array([0,   1  ])
        sweep=threshold_sweep(scores, labels, np.array([0.3, 0.7]))
        best=select_threshold(sweep, "max_f1")
        assert best in [0.3, 0.7]

    def test_bad_rule_raises_value_error(self):
        temp=[{"balanced_accuracy": 0.9, "f1": 0.8, "threshold": 0.5}]

        with pytest.raises(ValueError, match="Unknown"):
            select_threshold(temp,"min_error")


class TestRocData:

    def test_output_shapes(self):
        s=np.linspace(0, 1, 10)
        l=np.array([0,0,0,0,0,1,1,1,1,1])
        fprs,tprs=roc_data(s,l,np.linspace(0, 1, 20))

        assert fprs.shape==(20,) and tprs.shape==(20,)

    def test_fpr_tpr_in_range(self):
        rnad=np.random.default_rng(7)
        s=rnad.random(50)
        l=rnad.integers(0, 2, 50)
        fprs,tprs=roc_data(s,l,np.linspace(0, 1, 15))

        assert np.all((fprs>=0)&(fprs<=1))
        assert np.all((tprs>=0)&(tprs<=1))