from itertools import combinations
import pandas as pd
import json
import os

if __name__ == "__main__":
    # taken from here: https://github.com/Yushi-Hu/tifa/blob/main/human_annotations/human_annotations_with_scores.json
    with open('/mnt/qb/work/bethge/bkr405/data/compositionality-datasets/tifa_human_annotations.json', 'r') as f:
        js = json.load(f)
    j = [{**v, 'id': k} for k,v in js.items()]
    df = pd.json_normalize(j)
    rows = []
    prompts = df.text.unique()
    for prompt in prompts:
        images = df[df.text == prompt]
        for im1, im2 in combinations(range(len(images)), 2):
            im1_path = images.iloc[im1].image_path
            im2_path = images.iloc[im2].image_path
            im1_rating = images.iloc[im1].human_avg
            im2_rating = images.iloc[im2].human_avg
            if im1_rating == im2_rating:
                binary_rating = 0.5
            elif im1_rating > im2_rating:
                binary_rating = 1
            else:
                binary_rating = 0
            row = {
                'caption': prompt.strip('\r\n'),
                'caption_source': images.iloc[0].id.split('_')[0],
                'image_0_url': os.path.join('annotated_images', im1_path),
                'image_1_url': os.path.join('annotated_images', im2_path),
                'label_0': binary_rating,
                'label_1': 1 - binary_rating,
                'num_example_per_prompt': len(images),
                'model_0': '_'.join(images.iloc[im1].id.split('_')[2:]),
                'model_1': '_'.join(images.iloc[im2].id.split('_')[2:]),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"csvs/tifa.csv", index=False)