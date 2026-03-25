import argparse, json, os, sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pairs import generate_pairs, save_pairs

def main():
    a =argparse.ArgumentParser()
    a.add_argument("--config",required=True)
    args=a.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    splits_path = cfg["outputs"]["splits_path"]
    if not os.path.exists(splits_path):
        sys.exit(f"ERROR: {splits_path} not is not found. Please run the ingest_dataset.py first(refer README.md)")

    with open(splits_path) as f:
        splits = json.load(f)

    images_dir= cfg["data"]["images_dir"]
    pairs_v2_dir= cfg["outputs"]["pairs_dir"]+"_v2"
    dc= cfg.get("data_centric", {})
    min_imgs= dc.get("min_images_per_identity",2)
    max_imgs=dc.get("max_images_per_identity",10)

    print(f"filtering the min_images={min_imgs} and max_images={max_imgs}")
    print(f"Output:{pairs_v2_dir}/\n")

    for split_name, identity_list in splits.items():
        identity_paths,dropped = {}, 0
        for i in identity_list:
            person_dir = os.path.join(images_dir, i)
            if not os.path.isdir(person_dir):
                continue
            imgs = []
            files = os.listdir(person_dir)
            for j in files:
                if j.endswith(".jpg"):
                    full_path=os.path.join(person_dir, j)
                    imgs.append(full_path)
            imgs.sort()
            if len(imgs)<min_imgs:
                dropped = dropped + 1
                continue
            identity_paths[i]=imgs[0:max_imgs] 

        total=0
        for v in identity_paths.values():
            total = total + len(v)
        print(f"[{split_name}] {len(identity_list)} => {len(identity_paths)} identities "f"({dropped} dropped and {total} images kept)")
        seed = cfg["seed"]+abs(hash(split_name))%(2**31)
        pairs = generate_pairs(split_name, identity_paths,list(identity_paths.keys()),cfg["pairs"]["positive_per_split"],cfg["pairs"]["negative_per_split"],seed)
        save_pairs(pairs,os.path.join(pairs_v2_dir,f"{split_name}_pairs.csv"))

    print("\n filtered pair generation completed")

if __name__ == "__main__":
    main()