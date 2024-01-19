import pandas as pd
import os
from datasets import load_dataset, Dataset
from PIL import Image
from io import BytesIO


def _update(df, **kw):
    for k, v in kw.items():
        df[k] = v
    return df

df = pd.concat([
    _update(pd.read_csv("csvs/aigciqa2023.csv"), origin="aigciqa2023", split="train"),
    _update(pd.read_csv("csvs/eccv_caption.csv"), origin="eccv_caption", split="train"),
    _update(pd.read_csv("csvs/hpsv1_train.csv"), origin="hpsv1", split="train"),
    _update(pd.read_csv("csvs/hpsv1_test.csv"), origin="hpsv1", split="test"),
    _update(pd.read_csv("csvs/hpsv2_test.csv"), origin="hpsv2", split="test"),
    _update(pd.read_csv("csvs/image_reward_train.csv"), origin="image_reward", split="train"),
    _update(pd.read_csv("csvs/image_reward_validation.csv"), origin="image_reward", split="validation"),
    _update(pd.read_csv("csvs/image_reward_test.csv"), origin="image_reward", split="test"),
    _update(pd.read_csv("csvs/seetrue_train.csv"), origin="seetrue", split="train"),
    _update(pd.read_csv("csvs/seetrue_test.csv"), origin="seetrue", split="test"),
    _update(pd.read_csv("csvs/tifa.csv"), origin="tifa", split="train"),
])
# shuffle df
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

cols = ['caption', 'caption_source', 'image_0_url', 'image_1_url', 'label_0',
       'label_1', 'num_example_per_prompt', 'model_0', 'model_1', 'jpg_0',
       'jpg_1', 'are_different', 'has_label', 'origin', 'split']

def gen_dataset():
    # add pick-a-pic
    ds = load_dataset("yuvalkirstain/pickapic_v2", cache_dir="data/pickapic")
    for split in ("train", "validation", "test"):
        ds_split = ds[split]
        for i, row in zip(range(10000), ds_split):
            row_full = {}
            row["caption_source"] = "pickapic"
            row["origin"] = "pickapic"
            row["split"] = split
            for c in cols:
                row_full[c] = row[c]
            yield row_full
    # add other datasets
    for i, (_, row) in zip(range(10000), df.iterrows()):
        image_0_url = row.image_0_url
        image_1_url = row.image_1_url
        try:
            img_0 = Image.open(image_0_url)
            bytes0 = BytesIO()
            img_0.save(bytes0, format="JPEG")
            img_1 = Image.open(image_1_url)
            bytes_1 = BytesIO()
            img_1.save(bytes_1, format="JPEG")
        except Exception as e:
            print(e)
            continue
        row["jpg_0"] = bytes0.getvalue()
        row["jpg_1"] = bytes_1.getvalue()
        row["are_different"] = True
        row["has_label"] = True
        columns = row.keys()
        row_full = {}
        for c in cols:
            row_full[c] = row[c]
        yield row_full


ds = Dataset.from_generator(gen_dataset, cache_dir="data/dataset")
ds.push_to_hub("mehdidc/compositionality-subsample", token=True)