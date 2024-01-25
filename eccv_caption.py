from itertools import combinations
import pandas as pd
import json
import os

if __name__ == "__main__":
    # taken from here: hhttps://drive.google.com/drive/folders/1Sam8_Hpm4uWKB_Ehk9C_JcGNpYH7jZD2
    df = pd.read_csv('data/eccv_caption/mturk_parsed.csv')
    with open('data/eccv_caption/captions_val2014.json') as f:
        coco_captions_id_mapping = json.load(f)

    coco_id2cap = {}
    for c in coco_captions_id_mapping['annotations']:
        coco_id2cap[c['id']] = c['caption']

    duplicates = df.duplicated(subset=['cid', 'iid'], keep='first')
    df_filtered = df[~duplicates]
    df = df_filtered
    rows = []
    prompts = df.cid.unique()
    for prompt in prompts:
        curr_caption = coco_id2cap[prompt]
        images = df[df.cid == prompt]
        for im1, im2 in combinations(range(len(images)), 2):
            im1_path = 'COCO_val2014_{}.jpg'.format(str(images.iloc[im1].iid).zfill(12))
            im2_path = 'COCO_val2014_{}.jpg'.format(str(images.iloc[im2].iid).zfill(12))
            im1_rating = images.iloc[im1].annotation
            im2_rating = images.iloc[im2].annotation
            if im1_rating == im2_rating:
                binary_rating = 0.5
            # here a rating of 1 is better than a rating of 4
            elif im1_rating > im2_rating:
                binary_rating = 0
            else:
                binary_rating = 1
            row = {
                'caption': curr_caption,
                'caption_source': 'coco',
                'image_0_url': os.path.join('data', 'eccv_caption', 'val2014', im1_path),
                'image_1_url': os.path.join('data', 'eccv_caption', 'val2014', im2_path),
                'label_0': binary_rating,
                'label_1': 1 - binary_rating,
                'num_example_per_prompt': len(images),
                # TODO: better naming convention for natural images?
                'model_0': 'coco', # real images from COCO dataset
                'model_1': 'coco', # real images from COCO dataset
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"csvs/eccv_caption.csv", index=False)