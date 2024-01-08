import os
import pandas as pd
from itertools import combinations
from datasets import load_dataset
from clize import run

def main(*, root='.'):
    ds = load_dataset("yonatanbitton/SeeTRUE")
    for split in ('test',):
        df = ds.data[split].to_pandas()
        rows = []
        for text in df.text.unique():
            images = df[df.text == text]
            for im1, im2 in combinations(range(len(images)), 2):
                label1 = images.iloc[im1].label
                label2 = images.iloc[im2].label
                if label1 == label2:
                    binary_rating = 0.5
                elif label1 > label2:
                    binary_rating = 1
                else:
                    binary_rating = 0
                row = {
                    'caption': text,
                    'caption_source': images.iloc[0].dataset_source,
                    'image_0_url': images.iloc[im1].image["path"],
                    'image_1_url': images.iloc[im2].image["path"],
                    'label_0': binary_rating,
                    'label_1': 1 - binary_rating,
                    'num_example_per_prompt': len(images),
                    'model_0': '?',
                    'model_1': '?',
                }
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(f"seetrue_{split}.csv", index=False)

if __name__ == "__main__":
    run(main)