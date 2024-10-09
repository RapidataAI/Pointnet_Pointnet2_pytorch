import json
import os
import math
import matplotlib
import glob
import numpy as np
from matplotlib import patches

matplotlib.use('Agg')
from pathlib import Path
from typing import Tuple, List

import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
import itertools


YOLO_VAL_FOLDER = '/home/ubuntu/Pointnet_Pointnet2_pytorch/conv_nets/datasets/yolo_rapids_user_score_lines_128x128_us0.0/images/inference'
METADATA_FOLDER = '/home/ubuntu/Pointnet_Pointnet2_pytorch/conv_nets/datasets/metadata'
model_path = "/home/ubuntu/Pointnet_Pointnet2_pytorch/conv_nets/Points2BBox/user_score_lines_128_pretrained/weights/best.pt"
MODEL = YOLO(model_path)
COCO_PATH = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/coco'
BATCH_SIZE = 32
IMAGE_SIZE = 128
CONF=0.0
METADATA_DF = (pd.concat(pd.read_csv(os.path.join(METADATA_FOLDER, file)) for file in os.listdir(METADATA_FOLDER))
               .reset_index(drop=True).drop(columns='Unnamed: 0'))
OUTPUT_FOLDER = os.path.join('round2_images', f'eval_results_{CONF}_{IMAGE_SIZE}')


# Calculate IoU function
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    x2_min, y2_min, x2_max, y2_max = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def batched(iterable, n):

    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch

def get_label_for_rapid(rapid_id: str) -> Tuple[str, str, List[float]]:
    subset = METADATA_DF[METADATA_DF.rapid_ids==rapid_id]
    assert len(subset) == 1
    class_name = subset.iloc[0].class_names
    yolo_box = subset.iloc[0].yolo_boxes
    filename = subset.iloc[0].filenames
    return filename, class_name, json.loads(yolo_box)



def read_input_image(filename: str):
    path = os.path.join(YOLO_VAL_FOLDER, filename)
    return Image.open(path)

def read_original_image(image_filename: str):
    path = glob.glob(f'{COCO_PATH}/**/{image_filename}', recursive=True)[0]
    return Image.open(path)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    images = sorted(os.listdir(YOLO_VAL_FOLDER))
    prediction_results = []

    for batch_idx, batch in enumerate(tqdm(batched(images, BATCH_SIZE),
                                           total=int(math.ceil(len(images) / BATCH_SIZE)))):

        full_paths = [os.path.join(YOLO_VAL_FOLDER, file) for file in batch]
        rapid_ids = [Path(file).stem for file in batch]
        results = MODEL.predict(full_paths, conf=CONF, imgsz=IMAGE_SIZE)

        for result_index, result in enumerate(results):

            og_image_filename, class_name, yolo_gt = get_label_for_rapid(rapid_ids[result_index])

            original_image = read_original_image(og_image_filename)

            boxes = result.boxes
            xyxyn_boxes = [box.cpu().numpy().tolist() for box in boxes.xywhn]

            if not xyxyn_boxes:
                continue

            confs = [box.conf.item() for box in boxes]

            fig, ax = plt.subplots()
            ax.imshow(original_image)


            most_confident_box_idx = np.argmax(confs)
            best_box = xyxyn_boxes[most_confident_box_idx]

            gt_x_center, gt_y_center, gt_width, gt_height = yolo_gt


            gt_x = (gt_x_center - gt_width / 2) * original_image.width
            gt_y = (gt_y_center - gt_height / 2) * original_image.height
            gt_width *= original_image.width
            gt_height *= original_image.height

            gt_rect = patches.Rectangle((gt_x, gt_y), gt_width, gt_height, linewidth=1, edgecolor='g',
                                        facecolor='none', label='True')

            #ax.add_patch(gt_rect)

            x_center, y_center, width, height = best_box
            x = (x_center - width / 2) * original_image.width
            y = (y_center - height / 2) * original_image.height
            width *= original_image.width
            height *= original_image.height

            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            iou = calculate_iou((x,y, width, height), (gt_x, gt_y, gt_width, gt_height))

            ax.legend()

            plt.axis('off')
            folder = os.path.join(OUTPUT_FOLDER, class_name)
            os.makedirs(folder, exist_ok=True)
            plt.savefig(os.path.join(folder, f'{rapid_ids[result_index]}.png'), bbox_inches='tight')

            metadata = {
                'prediction': [x, y, width, height],
                'ground_truth': [gt_x, gt_y, gt_width, gt_height],
                'filename': og_image_filename,
                'class': class_name,
                'iou': iou,
                'used_training': model_path,
                'conf': confs[most_confident_box_idx]
            }

            with open(os.path.join(folder, f'{rapid_ids[result_index]}.json'), 'w') as f:
                json.dump(metadata, f)


if __name__ == '__main__':
    main()