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


YOLO_VAL_FOLDER = '/home/ubuntu/Pointnet_Pointnet2_pytorch/conv_nets/datasets/yolo_rapids_line_encode/images/val'
METADATA_FOLDER = '/home/ubuntu/Pointnet_Pointnet2_pytorch/conv_nets/datasets/metadata'
MODEL = YOLO("/home/ubuntu/Pointnet_Pointnet2_pytorch/conv_nets/yolo_rapids_line_encode/first_good/weights/best.pt")
COCO_PATH = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/coco'
BATCH_SIZE = 32
METADATA_DF = (pd.concat(pd.read_csv(os.path.join(METADATA_FOLDER, file)) for file in os.listdir(METADATA_FOLDER))
               .reset_index(drop=True).drop(columns='Unnamed: 0'))
OUTPUT_FOLDER = 'eval_results'


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
    all_ious = []
    for batch_idx, batch in enumerate(tqdm(batched(images, BATCH_SIZE),
                                           total=int(math.ceil(len(images)/BATCH_SIZE)))):

        full_paths = [os.path.join(YOLO_VAL_FOLDER, file) for file in batch]
        rapid_ids = [Path(file).stem for file in batch]
        results = MODEL(full_paths)

        for result_index, result in enumerate(results):

            og_image_filename, class_name, yolo_gt = get_label_for_rapid(rapid_ids[result_index])
            original_image = read_original_image(og_image_filename)
            model_input_image = read_input_image(batch[result_index])
            original_image_resized = original_image.resize(model_input_image.size)

            original_image_array = np.array(original_image_resized)
            model_input_image_array = np.array(model_input_image)

            mask = np.all(model_input_image_array == [0, 0, 0], axis=-1)
            alpha_channel = np.where(mask, 0, 1)
            model_input_image_with_alpha = np.dstack((model_input_image_array, alpha_channel * 255))

            fig, axes = plt.subplots(ncols=2)
            axes[0].imshow(original_image_array)
            axes[0].imshow(model_input_image_with_alpha, alpha=alpha_channel)
            axes[1].imshow(original_image)
            boxes = result.boxes
            xyxyn_boxes = [box.cpu().numpy().tolist() for box in boxes.xywhn]
            confs = [box.conf.item() for box in boxes]

            if not xyxyn_boxes:
                all_ious.append(-1)
                continue

            highest_conf_bbox = None


            for box_idx, box in enumerate(xyxyn_boxes):

                x_center, y_center, width, height = box
                x = (x_center - width / 2) * original_image_array.shape[1]
                y = (y_center - height / 2) * original_image_array.shape[0]
                width *= original_image_array.shape[1]
                height *= original_image_array.shape[0]

                rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none',
                                         label=f'Pred-conf{round(confs[box_idx],2)}')
                axes[0].add_patch(rect)

                if box_idx == np.argmax(confs):
                    highest_conf_bbox = [x, y, width, height]

            x_center, y_center, width, height = yolo_gt
            x = (x_center - width / 2) * original_image_array.shape[1]
            y = (y_center - height / 2) * original_image_array.shape[0]
            width *= original_image_array.shape[1]
            height *= original_image_array.shape[0]

            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='g', facecolor='none',
                                     label='True')
            axes[0].add_patch(rect)

            iou = calculate_iou(highest_conf_bbox, (x,y, width, height))
            all_ious.append(iou)

            axes[0].legend()
            plt.suptitle(class_name+ f'highest conf iou: {round(iou,2)}')
            plt.savefig(os.path.join(OUTPUT_FOLDER, f'{batch_idx}_{result_index}_preds.png'))
            plt.clf()
            plt.close()

    with open(os.path.join(OUTPUT_FOLDER, 'ious.json'), "w") as f:
        f.write(json.dumps(all_ious))



if __name__ == '__main__':
    main()