import numpy as np
import pytest
from src.validation import validate_pairs
from src.validation import validate_scores
from src.validation import validate_threshold
from src.validation import validate_config
from src.validation import validate_no_split_leakage

def make_pair(left="a.jpg", right="b.jpg", label="1", split="val"):
    return {"left_path": left, "right_path": right, "label": label, "split": split}


class TestValidatePairs:

    def test_basic_good_pairs(self):
        # one positive, one negative — should sail through
        pairs = [make_pair(label="1"), make_pair(label="0")]
        assert validate_pairs(pairs, check_paths=False)
    def test_empty_list_blows_up(self):
        with pytest.raises(ValueError, match="empty"):
            validate_pairs([], check_paths=False)
    def test_pair_without_split_key(self):
        bad = {"left_path": "a", "right_path": "b", "label": "1"}   # no 'split'
        with pytest.raises(ValueError, match="missing"):
            validate_pairs([bad], check_paths=False)

    def test_label_2_is_not_valid(self):
        with pytest.raises(ValueError, match="invalid label"):
            validate_pairs([make_pair(label="2")], check_paths=False)
    def test_unknown_split_name(self):
        with pytest.raises(ValueError, match="invalid split"):
            validate_pairs([make_pair(split="holdout")], check_paths=False)
    def test_integer_labels_also_work(self):
        pairs = [make_pair(label=0), make_pair(label=1)]
        assert validate_pairs(pairs, check_paths=False)


class TestValidateScores:

    def test_scores_match_pair_count(self):
        scores = np.array([0.5, 0.7])
        validate_scores(scores, [make_pair(), make_pair()])  
    def test_wrong_number_of_scores(self):
        with pytest.raises(ValueError, match="does not match"):
            validate_scores(np.array([0.5]), [make_pair(), make_pair()])



class TestValidateThreshold:

    def test_middle_of_range(self):
        validate_threshold(0.0)   
    def test_exact_boundary_values_are_ok(self):
        validate_threshold(-1.0)
        validate_threshold(1.0)
    def test_slightly_over_max(self):
        with pytest.raises(ValueError, match="outside allowed range"):
            validate_threshold(1.1)
    def test_way_under_min(self):
        with pytest.raises(ValueError, match="outside allowed range"):
            validate_threshold(-2.0)



class TestValidateConfig:

    def default_config(self):
        return {
            "evaluation": {
                "threshold_min":  -1.0,
                "threshold_max":   1.0,
                "metric":         "cosine",
                "threshold_rule": "max_balanced_accuracy",
            }
        }
    def test_happy_path(self):
        assert validate_config(self.default_config())
    def test_min_larger_than_max_raises(self):
        cfg = self.default_config()
        cfg["evaluation"]["threshold_min"] = 1.0  
        with pytest.raises(ValueError, match="threshold_min"):
            validate_config(cfg)

    def test_unsupported_metric(self):
        cfg = self.default_config()
        cfg["evaluation"]["metric"] = "l2"
        with pytest.raises(ValueError, match="Unknown metric"):
            validate_config(cfg)
    def test_unsupported_threshold_rule(self):
        cfg = self.default_config()
        cfg["evaluation"]["threshold_rule"] = "max_accuracy"
        with pytest.raises(ValueError, match="Unknown threshold_rule"):
            validate_config(cfg)
    def test_euclidean_is_a_valid_metric(self):
        cfg = self.default_config()
        cfg["evaluation"]["metric"] = "euclidean"
        assert validate_config(cfg)



class TestSplitLeakage:

    def test_totally_separate_splits(self):
        val_pairs  = [{"left_path":"a","right_path":"b"}]
        test_pairs = [{"left_path":"c","right_path": "d"}]
        assert validate_no_split_leakage(val_pairs, test_pairs)
    def test_same_pair_in_both_splits_is_leakage(self):
        shared = {"left_path":"a","right_path": "b"}
        with pytest.raises(ValueError, match="leakage"):
            validate_no_split_leakage([shared], [shared])