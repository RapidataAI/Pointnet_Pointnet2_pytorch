import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import tqdm
from PIL import Image
from pycocotools.coco import COCO


SPLIT = 'train'
filename = f'instances_{SPLIT}2017.json'
file = os.path.join('..', 'data', filename)
coco = COCO(file)

OUTPUT_DIR = os.path.join('..', 'data', 'coco', f'result_{filename.split(".")[0]}')

#train 20_000
#val 20_000
IMAGE_LIMIT = None

def download_image(img_url: str, img_paths_to_save: list):
    try:
        response = requests.get(img_url, stream=True)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))

        width, height = image.size

        for img_path in img_paths_to_save:
            with open(img_path, 'wb') as f:
                f.write(response.content)

        return img_url, (width, height)
    except Exception as e:
        return None, (None, None)


def normalize_bbox(width, height, bbox):
    return [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]


# Get all image ids
image_ids = coco.getImgIds()
image_ids = random.sample(image_ids, min(len(image_ids), (IMAGE_LIMIT or np.inf)))
categories = coco.loadCats(coco.getCatIds())
cat_lookup = {i['id']: i['name'] for i in categories}

dfs = []
images_to_download = defaultdict(list)

for img_id in tqdm.tqdm(image_ids):
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    if not anns:
        continue

    df = pd.DataFrame(anns)
    df.drop(columns=['segmentation', 'iscrowd'], inplace=True)
    im_data = coco.loadImgs([img_id])[0]

    df['file_name'] = im_data['file_name']
    df['image_url'] = im_data['coco_url']
    df['category_name'] = df.category_id.apply(lambda i: cat_lookup[i])
    dfs.append(df)

    category_counts = df.groupby('category_id').size().reset_index(name='count')
    only_one_cats = category_counts[category_counts['count'] == 1]
    df_with_unique_objects = df[df['category_id'].isin(only_one_cats['category_id'])]

    img_url = im_data['coco_url']
    img_name = im_data['file_name']

    for category_id in df_with_unique_objects.category_id.drop_duplicates().tolist():

        sub_df = df[df.category_id == category_id]
        cat_name = sub_df.iloc[0]['category_name']
        subdir = os.path.join(OUTPUT_DIR, cat_name)

        image_dir = os.path.join(subdir, 'images')
        os.makedirs(image_dir, exist_ok=True)

        img_path = os.path.join(image_dir, img_name)

        images_to_download[img_url].append(img_path)


all_labels = pd.concat(dfs)
all_labels.reset_index(drop=True, inplace=True)
print(f"Total images to download: {len(images_to_download)}")
failed_downloads = 0
image_dimensions = {}

with ThreadPoolExecutor(max_workers=100) as executor:

    results = [executor.submit(download_image, img_url, img_paths) for
               img_url, img_paths in images_to_download.items()]
    pbar = tqdm.tqdm(as_completed(results), total=len(results))

    for future in pbar:
        image_url, (width, height) = future.result()
        if image_url is None:
            failed_downloads += 1
            pbar.set_description(f'Failed downloads: {failed_downloads}')
            continue
        image_dimensions[image_url] = (width, height)



print(f"Total images failed to download: {failed_downloads} out of {len(images_to_download)}")



all_labels['bbox'] = all_labels.apply(
    lambda row: json.dumps(normalize_bbox(*image_dimensions.get(row.image_url, (1, 1)), row.bbox))#nojudge
    , axis=1
)

all_labels.to_csv(os.path.join(OUTPUT_DIR, f'{SPLIT}_all_labels.csv'))
print(all_labels.head())


