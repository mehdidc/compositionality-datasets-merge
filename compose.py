import pandas as pd
import os
from datasets import load_dataset, Dataset, DatasetDict
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

cols = ['caption', 'caption_source', 'image_0_url', 'image_1_url', 'label_0',
       'label_1', 'num_example_per_prompt', 'model_0', 'model_1', 'jpg_0',
       'jpg_1', 'are_different', 'has_label', 'origin', 'split']

def Infinite():
    i = 0
    while True:
        yield i
        i += 1

def gen_dataset(split, N=None):
    def gen():
        # add pick-a-pic
        ds = load_dataset("yuvalkirstain/pickapic_v2", cache_dir="data/pickapic")
        ds_split = ds[split]
        print(ds_split)
        it = range(N) if N else Infinite()
        for i, row in zip(it, ds_split):
            row_full = {}
            row["caption_source"] = "pickapic"
            row["origin"] = "pickapic"
            row["split"] = split
            for c in cols:
                row_full[c] = row[c]
            yield row_full
        # add other datasets
        df_split = df[df.split == split]
        it = range(N) if N else Infinite()
        for i, (_, row) in zip(it, df_split.iterrows()):
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
    return gen

train = Dataset.from_generator(gen_dataset("train"), cache_dir="data/dataset_train").shuffle(seed=42)
valid = Dataset.from_generator(gen_dataset("validation"), cache_dir="data/dataset_validation")
test = Dataset.from_generator(gen_dataset("test"), cache_dir="data/dataset_test")
ds = DatasetDict({"train": train, "validation": valid, "test": test})
ds.push_to_hub("mehdidc/compositionality-subsample", token=True)
