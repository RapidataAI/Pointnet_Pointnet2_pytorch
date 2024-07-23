import dataclasses
import os
import json
import re
from collections import defaultdict
from typing import List, Tuple, Dict

import imageio
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

WORKFLOW_DIR = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/exported_sessions'
COCO_IMAGES_ROOT = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/coco'
IMAGE_SIZE = (64, 64)
CLASS_REGEX = 'Paint the (.*?) with your finger! Be accurate!'
VALIDATION_SPLIT_LABELS = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/coco/result_instances_val2017/val_all_labels.csv'
TRAIN_SPLIT_LABELS = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/coco/result_instances_train2017/train_all_labels.csv'
USER_SCORE_THRESHOLD = 0.5
MODE = 'line_encode'
assert MODE in ['points_image', 'heatmap_image', 'line_encode'], "what kind of image do you want, Sir/Ma'am"

YOLO_FORMAT_OUTPUT_DIR = os.path.join('datasets', f'yolo_rapids_{MODE}')


def read_all_labels():
    val = pd.read_csv(VALIDATION_SPLIT_LABELS)
    train = pd.read_csv(TRAIN_SPLIT_LABELS)

    df = pd.concat([val, train]).reset_index(drop=True)

    return df


LABELS_DF = read_all_labels()


@dataclasses.dataclass(frozen=True)
class WorkflowResult:
    workflow_name: str
    images: List[np.ndarray]
    filenames: List[str]
    class_names: List[str]
    coco_boxes: List[List[float]]
    rapid_ids: List[str]

    def __post_init__(self):
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


def create_image(lines: List[List[Tuple[float, float]]]):
    points_channel = np.zeros(IMAGE_SIZE)
    point_idx_channel = np.zeros(IMAGE_SIZE)
    line_idx_channel = np.zeros(IMAGE_SIZE)

    for line_idx, line in enumerate(lines):
        for point_idx, (x, y) in enumerate(line):

            x, y = min(0.9999, x), min(0.9999, y)
            x, y = int(x * IMAGE_SIZE[0]), int(y * IMAGE_SIZE[1])

            points_channel[y, x] += 1
            line_idx_channel[y, x] = line_idx
            point_idx_channel[y, x] = point_idx

    points_channel = (points_channel / np.max(points_channel)) * 255
    line_idx_channel = (line_idx_channel / np.max(line_idx_channel)) * 255
    point_idx_channel = (point_idx_channel / np.max(point_idx_channel)) * 255

    if MODE == 'points_image':
        return points_channel
    elif MODE == 'heatmap_image':
        sigma = 1
        heatmap = scipy.ndimage.gaussian_filter(points_channel, sigma=sigma)
        return heatmap
    elif MODE == 'line_encode':
        return np.stack([points_channel, line_idx_channel, point_idx_channel], axis=-1)

    raise ValueError('Unknown image')


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
    prompt = guess['target']
    return re.match(CLASS_REGEX, prompt)[1].lower()


def get_original_filename_from_guess(guess: Dict):
    return guess['originalFileName']


def get_bounding_box_on_rapid(original_filename: str, class_name: str) -> List[float]:
    subset = LABELS_DF[(LABELS_DF.file_name == original_filename) & (LABELS_DF.category_name == class_name)]
    assert len(subset) == 1, 'Found less or more than 0 labels for a given file+class'
    return json.loads(subset.iloc[0].bbox)


def get_workflow_results(workflow_guesses: List[Dict]) -> WorkflowResult:
    rapid_points = defaultdict(list)
    rapid_original_images = dict()
    workflow_class = get_label_from_guess(workflow_guesses[0])
    workflow_id = workflow_guesses[0]['workflowId']['$oid']

    discarded = 0
    for guess in workflow_guesses:

        if guess['userScore'] < USER_SCORE_THRESHOLD:
            discarded += 1
            continue

        rapid_id = guess['rapidId']['$oid']

        lines = get_lines_from_guess(guess)
        rapid_points[rapid_id] += lines

        filename = get_original_filename_from_guess(guess)
        rapid_original_images[rapid_id] = filename

    print(f'DISCARDED {discarded}/{len(workflow_guesses)} guesses because of trust score')

    images = []
    filenames = []
    class_names = []
    bboxes = []
    rapids_ids = []
    for rapid_id, lines in rapid_points.items():
        filename = rapid_original_images[rapid_id]

        image = create_image(lines)

        bbox = get_bounding_box_on_rapid(filename, workflow_class)

        images.append(image)
        filenames.append(filename)
        class_names.append(workflow_class)
        bboxes.append(bbox)
        rapids_ids.append(rapid_id)

    return WorkflowResult(
        workflow_name=workflow_id,
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
    for split in ['train', 'val']:
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
                    os.path.join(METADATA_OUTPUT_DIR, f'{workflow_result.workflow_name}.csv'))


if __name__ == '__main__':
    main()
