import os
import json
import pandas as pd

if __name__ == "__main__":
    for split in ("train", "test"):
        path = f"data/hpsv1/preference_{split}.json"
        with open(path) as f:
            data = json.load(f)
        rows = []
        for r in data:
            index_preference = r["human_preference"]
            image_paths = r['file_path']
            caption = r['prompt']

            index_others = set(range(len(image_paths))) - set([index_preference])
            im1_path = (os.path.join("data", "hpsv1", image_paths[index_preference]))
            assert os.path.exists(im1_path), im1_path
            for index_other in index_others:
                im2_path = (os.path.join("data", "hpsv1", image_paths[index_other]))
                assert os.path.exists(im2_path)
                row = {
                    'caption': caption,
                    'caption_source': 'DiffusionDB',
                    'image_0_url': im1_path,
                    'image_1_url': im2_path,
                    'label_0': 1,
                    'label_1': 0,
                    'num_example_per_prompt': len(image_paths),
                    'model_0': 'stable-diffusion', #TODO missing model version
                    'model_1': 'stable-diffusion', #TODO missing model version
                }
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(f"csvs/hpsv1_{split}.csv", index=False)
