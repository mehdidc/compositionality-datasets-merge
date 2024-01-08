import os
import pandas as pd
from itertools import combinations
from datasets import load_dataset
from clize import run

def main(*, root='.'):
    for split in ('train', 'test',):
        if split == "test":
            ds = load_dataset("yonatanbitton/SeeTRUE")
            df = ds.data[split].to_pandas()
            df["image"] = df.image.apply(lambda x: x["path"])
        elif split == "train":
            # train split is not available in HF as a dataset, contrary to test
            df = pd.read_csv(os.path.join(root, "wysiwyr_train.csv"))
        else:
            raise ValueError(split)
        rows = []
        for text in df.text.unique():
            images = df[df.text == text]
            if len(images) == 1:
                row = {
                    'caption': text,
                    'caption_source': images.iloc[0].dataset_source,
                    'image_0_url': os.path.join(root, "wysiwyr_train_images", images.iloc[0].image),
                    'image_1_url': None,
                    'label_0': images.iloc[0].label,
                    'label_1': None,
                    'num_example_per_prompt': 1,
                    'model_0': '?',
                    'model_1': '?',
                }
                rows.append(row)
            else:
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
                        'image_0_url': images.iloc[im1].image,
                        'image_1_url': images.iloc[im2].image,
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