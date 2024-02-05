import pandas as pd
import json
import numpy as np

if __name__ == '__main__':

    with open('/mnt/qb/work/bethge/bkr405/data/compositionality-datasets/text2image_evaluation/prompt.txt') as f:
        prompts = [x.strip('\n') for x in f.readlines()]

    ratings = np.zeros((500, 2))
    for i in range(7):
        file_name = f"/mnt/qb/work/bethge/bkr405/data/compositionality-datasets/text2image_evaluation/data_{i}.json"
        with open(file_name, 'r') as f:
            ratings += np.array(json.load(f)[:500])
    ratings /= 7

    rows = []

    for prompt_id in range(500):

        img_ids = prompt_id * 2, prompt_id * 2 + 1
        chunk_1 = str(img_ids[0]).zfill(5)
        chunk_2 = str(img_ids[1]).zfill(5)
        img_path_1 = '/mnt/qb/work/bethge/bkr405/data/compositionality-datasets/text2image_evaluation/img/{}.jpg'.format(chunk_1)
        img_path_2 = '/mnt/qb/work/bethge/bkr405/data/compositionality-datasets/text2image_evaluation/img/{}.jpg'.format(chunk_2)

        im1_rating = ratings[prompt_id][0]
        im2_rating = ratings[prompt_id][1]

        if im1_rating == im2_rating:
            binary_rating = 0.5
        elif im1_rating > im2_rating:
            binary_rating = 1
        else:
            binary_rating = 0

        row = {
            'caption': prompts[prompt_id],
            'caption_source': 'DiffusionDB', # comes from ImageReward which internally uses DiffusionDB
            'image_0_url': img_path_1,
            'image_1_url': img_path_2,
            'label_0': binary_rating,
            'label_1': 1 - binary_rating,
            'num_example_per_prompt': 1,
            'model_0': 'stable-diffusion-v1.5',
            'model_1': 'd3po-stable-diffusion-v1.5',
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"csvs/d3po.csv", index=False)
