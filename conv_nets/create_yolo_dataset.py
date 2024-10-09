import dataclasses
import glob
import itertools
import os
import json
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional, Literal

import imageio
import numpy as np
import pandas as pd
import scipy
from PIL import Image
from tqdm import tqdm

WORKFLOW_DIR = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/exported_sessions'
COCO_IMAGES_ROOT = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/coco'
IMAGE_SIZE = (128, 128)
CLASS_REGEX = 'Paint the (.*?) with your finger! Be accurate!'
VALIDATION_SPLIT_LABELS = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/coco/result_instances_val2017/val_all_labels.csv'
TRAIN_SPLIT_LABELS = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/coco/result_instances_train2017/train_all_labels.csv'
USER_SCORE_THRESHOLD = 0.0

INFERENCE_ONLY_IMAGES = False



MODE = 'user_score_lines'
assert MODE in ['points_image', 'heatmap_image',
                'line_encode', 'line_encode_filter',
                'line_encode_and_draw', 'user_score_lines',
                'lines_on_channels'], "what kind of image do you want, Sir/Ma'am"


inference_suffix='_inference' if INFERENCE_ONLY_IMAGES else ''
YOLO_FORMAT_OUTPUT_DIR = (
    os.path.join('datasets',
                 f'custom_yolo_rapids_{MODE}_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}_us{USER_SCORE_THRESHOLD}{inference_suffix}'
                 )
)

def read_all_labels():
    val = pd.read_csv(VALIDATION_SPLIT_LABELS)
    train = pd.read_csv(TRAIN_SPLIT_LABELS)

    df = pd.concat([val, train]).reset_index(drop=True)

    return df


LABELS_DF = read_all_labels()

@dataclasses.dataclass(frozen=True)
class WorkflowResult:
    target_group_id: str
    images: List[np.ndarray]
    filenames: List[str]
    class_names: List[str]
    coco_boxes: List[List[float]]
    rapid_ids: List[str]

    def __post_init__(self):
        if not INFERENCE_ONLY_IMAGES:
            assert len(self.images) == len(self.filenames) == len(self.class_names) \
                   == len(self.coco_boxes) == len(self.rapid_ids), 'SUSSY LENGTHS'

            assert all(
                [0 < bbox[0] + bbox[2] <= 1 + 1e-6 for bbox in self.coco_boxes]), f'XYWH bboxes are weird {self.coco_boxes}'
            assert all(
                [0 < bbox[1] + bbox[3] <= 1 + 1e-6 for bbox in self.coco_boxes]), f'XYWH bboxes are weird {self.coco_boxes}'

    def __len__(self):
        return len(self.images)

    @property
    def yolo_bboxes(self):
        return [[x + w / 2, y + h / 2, w, h] for x, y, w, h in self.coco_boxes]

    def metadata_to_file(self, filename: str):
        df = pd.DataFrame(
            {
                'filenames': self.filenames,
                'class_names': self.class_names,
                'rapid_ids': self.rapid_ids,
                'coco_boxes': self.coco_boxes,
                'yolo_boxes': self.yolo_bboxes
            }
        )
        df.to_csv(filename)




def draw_bresenham_line(p1: Tuple[int, int], p2: Tuple[int, int], value: Any, image: np.ndarray,
                        agg: Literal["addition", "overwrite", "max"]='overwrite'):
    assert agg in ["addition", "overwrite", "max"], 'invalid param'
    x1, y1 = p1
    x2, y2 = p2

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    err = dx - dy

    while True:

        if agg == 'addition':
            image[y1, x1] += value
        elif agg == 'overwrite':
            image[y1, x1] = value
        elif agg == "max":
            image[y1, x1] = max(value, image[y1, x1])

        if x1 == x2 and y1 == y2:
            break

        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return image



def create_stacked_lines_image(lines: List[List[Tuple[float, float]]], line_user_scores: Optional[List[float]]):

    n_layers = 20
    layers = [np.zeros(IMAGE_SIZE) for _ in range(n_layers)]

    for line_idx, line in enumerate(lines):
        line_user_score = line_user_scores[line_idx]
        prev_x, prev_y = line[0]
        prev_x, prev_y = min(0.9999, prev_x), min(0.9999, prev_y)
        prev_x, prev_y = int(prev_x * IMAGE_SIZE[0]), int(prev_y * IMAGE_SIZE[1])

        for point_idx, (x, y) in enumerate(line):

            x, y = min(0.9999, x), min(0.9999, y)
            x, y = int(x * IMAGE_SIZE[0]), int(y * IMAGE_SIZE[1])

            draw_bresenham_line((prev_x, prev_y), (x, y), value=line_user_score,
                                image=layers[line_idx], agg='overwrite')

            prev_x, prev_y = x, y

    return np.stack(layers, axis=-1)

def create_image(lines: List[List[Tuple[float, float]]], line_user_scores: Optional[List[float]]):
    points_channel = np.zeros(IMAGE_SIZE)
    point_idx_channel = np.zeros(IMAGE_SIZE)
    line_idx_channel = np.zeros(IMAGE_SIZE)

    low_user_score = np.zeros(IMAGE_SIZE)
    medium_user_score = np.zeros(IMAGE_SIZE)
    high_user_score = np.zeros(IMAGE_SIZE)

    for line_idx, line in enumerate(lines):
        prev_x, prev_y = line[0]
        prev_x, prev_y = min(0.9999, prev_x), min(0.9999, prev_y)
        prev_x, prev_y = int(prev_x * IMAGE_SIZE[0]), int(prev_y * IMAGE_SIZE[1])

        for point_idx, (x, y) in enumerate(line):

            x, y = min(0.9999, x), min(0.9999, y)
            x, y = int(x * IMAGE_SIZE[0]), int(y * IMAGE_SIZE[1])



            if MODE == 'line_encode_and_draw':
                draw_bresenham_line((prev_x, prev_y), (x, y), 1.0, points_channel, agg='addition')
                draw_bresenham_line((prev_x, prev_y), (x, y), line_idx, line_idx_channel, agg='overwrite')
                draw_bresenham_line((prev_x, prev_y), (x, y), point_idx, point_idx_channel, agg='overwrite')

            elif MODE == 'user_score_lines':
                line_user_score = line_user_scores[line_idx]
                if line_user_score < 0.5:
                    draw_bresenham_line((prev_x, prev_y), (x, y), line_user_score, low_user_score, agg='max')
                elif 0.5 < line_user_score < 0.75:
                    draw_bresenham_line((prev_x, prev_y), (x, y), line_user_score, medium_user_score, agg='max')
                else:
                    draw_bresenham_line((prev_x, prev_y), (x, y), line_user_score, high_user_score, agg='max')
            else:
                points_channel[y, x] += 1.0
                line_idx_channel[y, x] = line_idx
                point_idx_channel[y, x] = point_idx

            prev_x, prev_y = x, y

    if MODE != 'user_score_lines':
        points_channel = (points_channel / np.max(points_channel)) * 255
        line_idx_channel = (line_idx_channel / np.max(line_idx_channel)) * 255
        point_idx_channel = (point_idx_channel / np.max(point_idx_channel)) * 255
    else:
        low_user_score = (low_user_score / np.max(low_user_score)) * 255
        medium_user_score = (medium_user_score / np.max(medium_user_score)) * 255
        high_user_score = (high_user_score / np.max(high_user_score)) * 255


    if MODE == 'user_score_lines':
        return np.stack([high_user_score, medium_user_score, low_user_score], axis=-1)

    if MODE == 'points_image':
        return points_channel

    elif MODE == 'heatmap_image':
        sigma = 1
        heatmap = scipy.ndimage.gaussian_filter(points_channel, sigma=sigma)
        return heatmap
    elif MODE in ['line_encode', 'line_encode_filter', 'line_encode_and_draw']:
        return np.stack([points_channel, line_idx_channel, point_idx_channel], axis=-1)

    raise ValueError('Unknown mode')


def get_lines_from_guess(guess: Dict) -> List[List[Tuple[float, float]]]:
    all_lines = []
    for line in guess['result']['lines']:
        points = []
        for point in line['points']:
            x, y = float(point['x']), float(point['y'])
            points.append((x, y))
        all_lines.append(points)
    return all_lines


def get_label_from_guess(guess: Dict):
    if INFERENCE_ONLY_IMAGES:
        return 'UNKNOWN'
    prompt = guess['target']
    return re.match(CLASS_REGEX, prompt)[1].lower()


def get_original_filename_from_guess(guess: Dict):
    if INFERENCE_ONLY_IMAGES:
        return 'UNKNOWN'
    return guess['originalFileName']


def get_coco_bounding_box_on_rapid(original_filename: str, class_name: str) -> List[float]:
    if INFERENCE_ONLY_IMAGES:
        return [0,0,0,0]
    subset = LABELS_DF[(LABELS_DF.file_name == original_filename) & (LABELS_DF.category_name == class_name)]
    assert len(subset) == 1, 'Found less or more than 0 labels for a given file+class'
    return json.loads(subset.iloc[0].bbox)


def filter_lines_with_medians(lines: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:

    points = list(itertools.chain(*lines))
    sum_points_og = len(points)
    points = np.array(points)

    medians = np.median(points, axis=0)

    mad = np.median(np.abs(points - medians), axis=0)

    threshold = 2.5 * mad

    filtered_lines = []
    filtered_points = 0
    for line in lines:
        filtered_line = [point for point in line if np.all(np.abs(point - medians) <= threshold)]
        filtered_points += len(filtered_line)
        filtered_lines.append(filtered_line)

    print(f'REMOVED {sum_points_og-filtered_points}/{sum_points_og} points because they were too far from the median')
    return filtered_lines



def get_workflow_results(workflow_guesses: List[Dict]) -> WorkflowResult:
    rapid_points = defaultdict(list)
    user_scores = defaultdict(list)
    rapid_original_images = dict()
    workflow_class = get_label_from_guess(workflow_guesses[0])
    target_group_id = workflow_guesses[0]['targetGroupId']['$oid']

    discarded_because_user_score = 0
    for guess in workflow_guesses:
        user_score = guess['userScore']
        if user_score < USER_SCORE_THRESHOLD:
            discarded_because_user_score += 1
            continue

        rapid_id = guess['rapidId']['$oid']

        lines = get_lines_from_guess(guess)
        rapid_points[rapid_id] += lines
        user_scores[rapid_id] += [user_score]*len(lines)

        filename = get_original_filename_from_guess(guess)
        rapid_original_images[rapid_id] = filename
    if discarded_because_user_score:
        print(f'DISCARDED {discarded_because_user_score}/{len(workflow_guesses)} guesses because of trust score')

    images = []
    filenames = []
    class_names = []
    bboxes = []
    rapids_ids = []
    discarded_because_obj_size = 0
    for rapid_id, lines in rapid_points.items():

        if MODE == 'line_encode_filter':
            lines = filter_lines_with_medians(lines)

        line_user_scores = user_scores[rapid_id]
        image = create_image(lines, line_user_scores)

        filename = rapid_original_images[rapid_id]

        bbox = get_coco_bounding_box_on_rapid(filename, workflow_class)


        images.append(image)
        filenames.append(filename)
        class_names.append(workflow_class)
        bboxes.append(bbox)
        rapids_ids.append(rapid_id)

    print(f'DISCARDED {discarded_because_obj_size}/{len(rapid_points)} {class_names[0]} rapids because of object size')
    return WorkflowResult(
        target_group_id=target_group_id,
        images=images,
        filenames=filenames,
        class_names=class_names,
        coco_boxes=bboxes,
        rapid_ids=rapids_ids
    )


def read_workflows(exported_workflows_dir: str) -> List[WorkflowResult]:
    for workflow_guesses_file in tqdm(os.listdir(exported_workflows_dir)):
        with open(os.path.join(exported_workflows_dir, workflow_guesses_file), 'r') as f:
            workflow_guesses = json.load(f)

        workflow_result = get_workflow_results(workflow_guesses)
        yield workflow_result


def main():
    splits = ['inference'] if INFERENCE_ONLY_IMAGES else ['train', 'val']
    for split in splits:
        split_dir = os.path.join(WORKFLOW_DIR, split)

        OUTPUT_LABEL_DIR = os.path.join(YOLO_FORMAT_OUTPUT_DIR, 'labels', split)
        OUTPUT_IMAGE_DIR = os.path.join(YOLO_FORMAT_OUTPUT_DIR, 'images', split)
        METADATA_OUTPUT_DIR = os.path.join(YOLO_FORMAT_OUTPUT_DIR, 'metadata')
        DEFAULT_YOLO_CLASS = 0

        os.makedirs(YOLO_FORMAT_OUTPUT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
        os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
        os.makedirs(METADATA_OUTPUT_DIR, exist_ok=True)

        for workflow_result in read_workflows(split_dir):
            yolo_boxes = workflow_result.yolo_bboxes
            for idx in range(len(workflow_result)):
                image = workflow_result.images[idx]
                rapid_id = workflow_result.rapid_ids[idx]
                yolo_box = yolo_boxes[idx]
                with open(os.path.join(OUTPUT_LABEL_DIR, f'{rapid_id}.txt'), 'w') as f:
                    f.write(f'{DEFAULT_YOLO_CLASS} {" ".join([str(c) for c in yolo_box])}\n')
                imageio.imwrite(os.path.join(OUTPUT_IMAGE_DIR, f'{rapid_id}.png'), image.astype(np.uint8))
                workflow_result.metadata_to_file(
                    os.path.join(METADATA_OUTPUT_DIR, f'{workflow_result.target_group_id}.csv'))


if __name__ == '__main__':
    main()
