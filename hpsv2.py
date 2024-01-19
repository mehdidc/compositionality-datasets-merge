import os
import json
import pandas as pd
from itertools import combinations

if __name__ == "__main__":
    for split in ("train", "test"):
        path = os.path.join("data", "hpsv2", f"{split}.json")
        if not os.path.exists(path):
            print(path, "not found. Skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        rows = []
        for r in data:
            ranks = r["rank"]
            image_paths = r['image_path']
            caption = r['prompt']
            for im1, im2 in combinations(range(len(image_paths)), 2):
                im1_rank = ranks[im1]
                im2_rank = ranks[im2]
                binary_rating = im1_rank < im2_rank
                im1_path = (os.path.join("data", "hpsv2", split, image_paths[im1]))
                im2_path = (os.path.join("data", "hpsv2", split, image_paths[im2]))
                assert os.path.exists(im1_path)
                assert os.path.exists(im2_path)
                row = {
                    'caption': r["prompt"],
                    'caption_source': '?', #TODO missing caption source
                    'image_0_url': im1_path,
                    'image_1_url': im2_path,
                    'label_0': binary_rating,
                    'label_1': 1 - binary_rating,
                    'num_example_per_prompt': len(image_paths),
                    'model_0': '?', #TODO missing model version
                    'model_1': '?', #TODO missing model version
                }
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(f"csvs/hpsv2_{split}.csv", index=False)

