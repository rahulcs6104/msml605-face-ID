import os


VALID_LABELS = {"0", "1", 0, 1}
VALID_SPLITS = {"train", "val", "test"}
NEEDED_COLUMNS = {"left_path", "right_path", "label", "split"}


def validate_pairs(pairs: list, check_paths: bool = True):
    if not pairs:
        raise ValueError("The pair list is empty.")
    for i, pair in enumerate(pairs):
        missing =NEEDED_COLUMNS-set(pair.keys())
        if missing:
            raise ValueError(f"Pair {i} is missing columns {missing}")
        if pair["label"] not in VALID_LABELS and str(pair["label"]) not in VALID_LABELS:
            raise ValueError(f"Pair {i} has invalid label '{pair['label']}' => must be either 0 or 1")
        if pair["split"] not in VALID_SPLITS:
            raise ValueError(f"Pair {i}is invalid split '{pair['split']}' => must be one of train/val/test")
        if check_paths:
            for key in ("left_path", "right_path"):
                if not os.path.exists(pair[key]):
                    raise ValueError(f"Pair {i}: path does not exist — {pair[key]}")
    return True

def validate_scores(scores,pairs: list):
    if len(scores)!=len(pairs):
        raise ValueError(f"ERROR => Score count ({len(scores)}) does not match pair count ({len(pairs)})")
    return True



def validate_threshold(threshold:float,min_val:float = -1.0,max_val:float = 1.0):
    if not (min_val<=threshold<=max_val):
        raise ValueError(f"the threshold {threshold} is outside allowed range  of [{min_val}, {max_val}]")
    return True


def validate_config(cfg:dict):
    e = cfg.get("evaluation", {})
    threshold_min = e.get("threshold_min", -1.0)
    threshold_max = e.get("threshold_max",  1.0)
    if threshold_min >= threshold_max:
        raise ValueError(f"threshold_min ({threshold_min}) must be less than the threshold_max ({threshold_max})")
    if e.get("metric", "cosine") not in ("cosine", "euclidean"):
        raise ValueError(f"Unknown metric '{e.get('metric')}' =>please use cosine or euclidean")
    if e.get("threshold_rule","") not in ("max_balanced_accuracy","max_f1"):
        raise ValueError(f"Unknown threshold_rule '{e.get('threshold_rule')}'")
    return True




def validate_no_duplicate(val_pairs:list,test_pairs:list):# to make sure same pairs dont appear in validate and test
    def key(p):
        return (p["left_path"], p["right_path"])
    overlap = {key(p) for p in val_pairs} & {key(p) for p in test_pairs}
    if overlap:
        raise ValueError(f"Split leakage detected: {len(overlap)} pair(s) in both val and test.")
    return True