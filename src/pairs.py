
# Each pair: (left_path, right_path, label, split)
# in each pair we will have a left image and right image
# if label-> 1 (then both of images matches)
# if label-> 0 (then both the images dont match)
# split tell us which one of the split it is from (train , validate or test)


import csv
from pathlib import Path

import numpy as np


def generate_pairs(split_name: str,identity_paths: dict,identity_list: list,n_pos: int,n_neg: int,seed: int):

    rng = np.random.default_rng(seed)
    identity_list = sorted(identity_list)
    pairs = []

    #postive pairs (the left and the right image will be from the same person)
    pos_candidates = []
    for person in identity_list:           # renamed to person
        imgs = sorted(identity_paths[person])
        for i in range(len(imgs)):         # i is now safe to use here
            for j in range(i + 1, len(imgs)):
                pos_candidates.append((imgs[i], imgs[j]))

    idx = np.arange(len(pos_candidates))
    rng.shuffle(idx)
    for k in idx[:n_pos]:
        left, right = pos_candidates[k]
        pairs.append({
            "left_path": left,
            "right_path": right,
            "label": 1,
            "split": split_name,
        })

    #negative pairs (the left and the right image will be from the different persons)
    n_ids = len(identity_list)
    visited = set()
    neg_count = 0

    while neg_count < n_neg:
    
        i,j = rng.integers(0, n_ids,size=2)
        if i==j:
            continue
        key=(min(i, j),max(i, j))
        if key in visited:
            continue
        visited.add(key)
        index_a = identity_list[key[0]]
        img_a = sorted(identity_paths[index_a])
        index_b = identity_list[key[1]]
        img_b = sorted(identity_paths[index_b])
        pairs.append({"left_path":img_a[0],"right_path":img_b[0],  "label":0,"split":split_name})
        neg_count =neg_count + 1
    return pairs

#this function writes the pairs to a CSV
def save_pairs(pairs: list, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["left_path", "right_path", "label", "split"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)
    print(f"{len(pairs)} pairs saved in => {output_path}")

#this fucntion is to load the csv into a dicts
def load_pairs(path: str):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)