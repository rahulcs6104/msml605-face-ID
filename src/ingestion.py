import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow_datasets as tfds


def load_and_save_dataset(tfds_name: str,cache_dir:str, images_dir:str) -> dict:
    print("loading the dataset")
    images_dir = Path(images_dir)
    ds,info = tfds.load(tfds_name,split="train",with_info=True,data_dir=cache_dir,shuffle_files=False,)
    #create persons like eg: rahul
    person: dict = {}
    for global_idx, example in enumerate(ds):
        name = example["label"].numpy().decode("utf-8")
        img =  example["image"].numpy()
        person.setdefault(name,[]).append((global_idx,img))
    #put each persons image into their respective directories
    i_paths: dict = {}
    for name in sorted(person.keys()):                         
        items = sorted(person[name], key=lambda x: x[0])      
        identity_dir = images_dir / name
        identity_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for local_idx, (_, img_arr) in enumerate(items):
            fname = f"{local_idx:04d}.jpg"
            fpath = identity_dir/fname
            if not fpath.exists():                          
                Image.fromarray(img_arr).save(str(fpath), format="JPEG", quality=95)
            paths.append(str(fpath))                        
        i_paths[name] = paths

    print(f"saved images for this person ({len(i_paths)})  to ({images_dir}) directory")
    return i_paths


def split_identities(i_paths: dict, train_ratio: float, val_ratio: float, seed: int) -> dict:
    sorted_ids = sorted(i_paths.keys())
    total_num = len(sorted_ids)

    rng = np.random.default_rng(seed)
    indices = np.arange(total_num)
    rng.shuffle(indices)          #will be able to recreate as the seed is set in one of the folder

    train_num = int(total_num *train_ratio)
    val_num = int(total_num*val_ratio)

    train_ids = []
    for i in indices[:train_num]:
        train_ids.append(sorted_ids[i])
    val_ids =[]
    for i in indices[train_num:train_num+val_num]:
        val_ids.append(sorted_ids[i])

    test_ids= []
    for i in indices[train_num + val_num:]:
        test_ids.append(sorted_ids[i])
    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids}
    return splits

def write_manifest(
    manifest_path: str,
    splits_path: str,
    seed: int,
    split_policy: str,
    splits: dict,
    i_paths: dict,
    data_source: dict,
) -> None:
    counts = {}
    for split_name, identities in splits.items():
        counts[split_name] = {
            "identities": len(identities),
            "images": sum(len(i_paths[iid]) for iid in identities),
        }
    manifest = {
        "seed":seed,
        "split_policy":split_policy,
        "counts":counts,
        "data_source":data_source,
    }
    #for manifest file
    Path(manifest_path).parent.mkdir(parents=True, exist_ok= True)
    with open(manifest_path,"w") as f:
        json.dump(manifest,f,indent=2)
    print(f"manifest file created and called ==> {manifest_path}")

    #for splits file
    Path(splits_path).parent.mkdir(parents=True,exist_ok= True)
    with open(splits_path,"w") as f:
        json.dump(splits,f,indent=2)
    print(f"splits file created and called ==> {splits_path}")