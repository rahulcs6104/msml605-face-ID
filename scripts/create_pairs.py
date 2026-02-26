import argparse
import json
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pairs import generate_pairs, save_pairs

def main():
    parser = argparse.ArgumentParser(description="generate verification pairs.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file(m1.yaml)")
    args = parser.parse_args()

    with open(args.config) as f:
        config_values = yaml.safe_load(f)

    splits_path = config_values["outputs"]["splits_path"]
    if not Path(splits_path).exists():#if the splits.json file doesnt exist
        print(f"ERRROR => {splits_path} is not to be found. please run ingest_dataset.py file first to have the splits.json")
        sys.exit(1)
    with open(splits_path) as f:
        splits = json.load(f) 

    #load manifest to get images_dir 
    with open(config_values["outputs"]["manifest_path"]) as f:
        manifest = json.load(f)

    images_dir = manifest["data_source"]["images_dir"]
    i_paths = {}
    images_dir_path = Path(images_dir)

    for i in sorted(images_dir_path.iterdir()):
        if i.is_dir():
            imgs = sorted(str(p) for p in i.glob("*.jpg"))
            i_paths[i.name] = imgs

    print(f"loaded {len(i_paths)} identities from {images_dir}")
    n_pos = config_values["pairs"]["positive_per_split"]
    n_neg = config_values["pairs"]["negative_per_split"]
    seed  = config_values["seed"]
    pairs_dir = config_values["outputs"]["pairs_dir"]

    for split_name, identity_list in splits.items():
        split_seed = seed + hash(split_name) % 1000

        pairs = generate_pairs(split_name=split_name,i_paths=i_paths,identity_list=identity_list,n_pos=n_pos,n_neg=n_neg,seed=split_seed,)
        out_path =Path(pairs_dir)/f"{split_name}_pairs.csv"
        save_pairs(pairs,str(out_path))

    print("create_pairs done")


if __name__ == "__main__":
    main()