import os
import json
import pandas as pd
from clize import run

def main(*, root='.'):
    for split in ("train", "test"):
        path = os.path.join(root, f"preference_{split}.json")
        with open(path) as f:
            data = json.load(f)
        rows = []
        for r in data:
            index_preference = r["human_preference"]
            image_paths = r['file_path']
            caption = r['prompt']

            index_others = set(range(len(image_paths))) - set([index_preference])
            for index_other in index_others:
                row = {
                    'caption': caption,
                    'caption_source': 'DiffusionDB',
                    'image_0_url': os.path.abspath(os.path.join(root, "preference_images", image_paths[index_preference])),
                    'image_1_url': os.path.abspath(os.path.join(root,  "preference_images", image_paths[index_other])),
                    'label_0': 1,
                    'label_1': 0,
                    'num_example_per_prompt': len(image_paths),
                    'model_0': 'stabilityai/stable-diffusion-?-?',
                    'model_1': 'stabilityai/stable-diffusion-?-?',
                }
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(f"hpsv1_{split}.csv", index=False)

if __name__ == "__main__":
    run(main)
