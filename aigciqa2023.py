from itertools import combinations
import pandas as pd
import os

if __name__ == "__main__":
    # taken from here: https://github.com/wangjiarui153/AIGCIQA2023/pull/4/files
    df = pd.read_csv("data/aigciqa2023/AIGCIQA2023.csv")
    models = pd.read_excel("data/aigciqa2023/pic-index.xlsx", header=None).iloc[:, 0].to_list()
    rows = []
    prompts = df.prompt.unique()
    for prompt in prompts:
        images = df[df.prompt == prompt]
        for im1, im2 in combinations(range(len(images)), 2):
            im1_path = str(images.iloc[im1].name)+'.png'
            im2_path = str(images.iloc[im2].name)+'.png'
            # TODO: for now only taking image-text alignment scores, ignoring quality and authenticity scores
            # TODO: ignoring standard deviations, only taking means
            im1_rating = images.iloc[im1].correspondence_mos
            im2_rating = images.iloc[im2].correspondence_mos
            if im1_rating == im2_rating:
                binary_rating = 0.5
            elif im1_rating > im2_rating:
                binary_rating = 1
            else:
                binary_rating = 0
            im1_path = os.path.join('data', 'aigciqa2023', 'allimg', im1_path)
            im2_path = os.path.join('data', 'aigciqa2023', 'allimg', im2_path)
            assert os.path.exists(im1_path)
            assert os.path.exists(im2_path)
            row = {
                'caption': prompt,
                'caption_source': 'PartiPrompts',
                'image_0_url': im1_path,
                'image_1_url': im2_path,
                'label_0': binary_rating,
                'label_1': 1 - binary_rating,
                'num_example_per_prompt': len(images),
                'model_0': models[int(images.iloc[im1].name)].replace("stable_diffusion", "stable-diffusion"),
                'model_1': models[int(images.iloc[im2].name)].replace("stable_diffusion", "stable-diffusion"),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"csvs/aigciqa2023.csv", index=False)