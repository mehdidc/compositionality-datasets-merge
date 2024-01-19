from itertools import combinations
import pandas as pd
from datasets import load_dataset


if __name__ == "__main__":
    # referring to https://github.com/THUDM/ImageReward/issues/26
    ds = load_dataset("THUDM/ImageRewardDB", "8k")
    print(ds)
    for split in ("train", "validation", "test"):
        df = ds.data[split].to_pandas()
        rows = []
        for prompt_id in df.prompt_id.unique():
            images = df[df.prompt_id == prompt_id]
            for im1, im2 in combinations(range(len(images)), 2):
                im1_path = (images.iloc[im1].image['path'])
                im2_path = (images.iloc[im2].image['path'])
                im1_rating = images.iloc[im1].overall_rating
                im2_rating = images.iloc[im2].overall_rating
                if im1_rating == im2_rating:
                    binary_rating = 0.5
                elif im1_rating > im2_rating:
                    binary_rating = 1
                else:
                    binary_rating = 0
                row = {
                    'caption': images.iloc[0].prompt,
                    'caption_source': 'DiffusionDB',
                    'image_0_url': im1_path,
                    'image_1_url': im2_path,
                    'label_0': binary_rating,
                    'label_1': 1 - binary_rating,
                    'num_example_per_prompt': len(images),
                    # TODO not sure which Stable Diffision model is used in DiffusionDB??
                    'model_0': 'stable-diffusion', #TODO missing model version
                    'model_1': 'stable-diffusion', #TODO missing model version
                }
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(f"csvs/image_reward_{split}.csv", index=False)
