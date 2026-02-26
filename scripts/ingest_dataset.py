import argparse
import sys
from pathlib import Path
import yaml

sys.path.insert(0,str(Path(__file__).parent.parent))
from src.ingestion import (load_and_save_dataset,split_identities,write_manifest,)

def main():
    parser = argparse.ArgumentParser(description="ingest lwf daataset")
    parser.add_argument("--config", required=True, help="please give path to YAML config file, if it is in config folder type --config while running")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg=yaml.safe_load(f)

    #loading images from the using a function from ingestion.py
    identity_paths = load_and_save_dataset(
        tfds_name=cfg["data"]["tfds_name"],
        cache_dir=cfg["data"]["cache_dir"],
        images_dir=cfg["data"]["images_dir"],
    )

    #dividing identities
    splits = split_identities(
        i_paths=identity_paths,
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"],
    )
    for split_name in splits:
        ids = splits[split_name]
        images_num = 0
        for i in ids:
            images_num+=len(identity_paths[i])
        print(split_name,":",len(ids),"identities,", images_num, "images")

    write_manifest(
        manifest_path=cfg["outputs"]["manifest_path"],
        splits_path=cfg["outputs"]["splits_path"],
        seed=cfg["seed"],
        split_policy=cfg["split_policy"],
        splits=splits,
        i_paths=identity_paths,
        data_source={
            "tfds_name": cfg["data"]["tfds_name"],
            "cache_dir": cfg["data"]["cache_dir"],
            "images_dir": cfg["data"]["images_dir"],
        },
    )
    print("done loading and saved the dataset (check the data folder)")
if __name__ == "__main__":
    main()