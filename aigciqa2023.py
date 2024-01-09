from itertools import combinations
import pandas as pd

if __name__ == "__main__":
    # taken from here: https://github.com/wangjiarui153/AIGCIQA2023/pull/4/files
    df = pd.read_csv('/mnt/qb/work/bethge/bkr405/data/compositionality-datasets/AIGCIQA2023.csv')
    models = pd.read_csv('/mnt/qb/work/bethge/bkr405/data/compositionality-datasets/pic-index.csv', header=None).iloc[:, 0].to_list()
    rows = []
    prompts = df.prompt.unique()
    for prompt in prompts:
        images = df[df.prompt == prompt]
        for im1, im2 in combinations(range(len(images)), 2):
            im1_path = (images.iloc[im1].name)
            im2_path = (images.iloc[im2].name)
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
            row = {
                'caption': prompt,
                'caption_source': 'PartiPrompts',
                'image_0_url': im1_path,
                'image_1_url': im2_path,
                'label_0': binary_rating,
                'label_1': 1 - binary_rating,
                'num_example_per_prompt': len(images),
                'model_0': models[im1],
                'model_1': models[im2],
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"aigciqa2023.csv", index=False)