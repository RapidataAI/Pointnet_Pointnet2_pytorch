import glob
import itertools
import json
import os
import random
import re
from typing import List, Tuple, Any

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

POINTS_PER_RAPID = 1024


class LineRapidDataLoader(Dataset):

    points_arr_file = 'points.npy'
    bboxes_arr_file = 'bboxes.npy'

    def __init__(self, points: np.ndarray, bboxes: np.ndarray):
        # expected array: images x nPoints x 3
        assert len(points) == len(bboxes)

        assert len(points.shape) == 3
        assert points.shape[-1] == 3
        self.points = points

        assert len(bboxes.shape) == 2
        assert bboxes.shape[-1] == 4
        self.bboxes = bboxes

    @classmethod
    def from_directory(cls, directory: str):
        points = np.load(os.path.join(directory, cls.points_arr_file))
        bboxes = np.load(os.path.join(directory, cls.bboxes_arr_file))
        return LineRapidDataLoader(points, bboxes)

    def to_directory(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, self.points_arr_file), self.points)
        np.save(os.path.join(directory, self.bboxes_arr_file), self.bboxes)
        return self

    @classmethod
    def concat_dataset_from_root(cls, directory: str):
        datasets = []
        for workflow_id in os.listdir(directory):
            dataset = LineRapidDataLoader.from_directory(os.path.join(directory, workflow_id))
            datasets.append(dataset)
        return torch.utils.data.ConcatDataset(datasets)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return torch.from_numpy(self.points[index]), torch.from_numpy(self.bboxes[index])


def read_grountruths_under_folder(folder: str, split: str) -> pd.DataFrame:

    hits = glob.glob(f'{folder}/**/{split}_all_labels.csv', recursive=True)
    assert len(hits) == 1, 'Multiple or 0 hits found for labels.csv'

    df = pd.read_csv(hits[0])

    df.drop_duplicates(subset=['file_name', 'category_name'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def get_groundtruth_for(filename: str, object_class: str, groundtruth_df: pd.DataFrame):
    row = groundtruth_df[
        (groundtruth_df.file_name == filename)
        & (groundtruth_df.category_name == object_class)
        ].iloc[0]

    bbox_arr = json.loads(row.bbox)
    return bbox_arr


def get_lines(rapid_doc) -> List[List[Tuple[float, float]]]:
    lines = []
    for guess in rapid_doc['guesses']:
        for line in guess['result']['lines']:
            coords = []
            for point in line['points']:
                x, y = float(point['x']), float(point['y'])
                coords.append((x, y))
            lines.append(coords)

    return lines


def pad_or_clip_list(points: List[List[Tuple[float, float]]], target_length: int, pad_value: Any):
    if len(points) < target_length:
        points = points + [pad_value] * (target_length - len(points))
    elif len(points) > target_length:
        points = random.sample(points, target_length)

    assert len(points) == target_length
    return points


def read_points_from_workflow(workflow_rapids_json: str):
    with open(workflow_rapids_json, 'r') as f:
        rapids = json.load(f)

    points_array_for_rapids = []
    for rapid in rapids:
        lines = get_lines(rapid)

        points = list(itertools.chain(*lines))
        points = pad_or_clip_list(points, target_length=POINTS_PER_RAPID, pad_value=(0, 0))
        points = np.array(points)
        z_coords = np.zeros((points.shape[0], 1))
        points = np.hstack((points, z_coords))

        points_array_for_rapids.append(points)

    points = np.array(points_array_for_rapids)
    assert points.shape == (len(rapids), POINTS_PER_RAPID, 3), f'Suspicious shape {points.shape}'
    assert np.min(points) >= 0 and np.max(points) <= 1, f'Suspicious values. min: {np.min(points)} max {np.max(points)}'
    return points

def read_labels_from_workflow(workflow_rapids_json: str, groundtruths: pd.DataFrame,
                              prompt_to_class_pattern: str):

    with open(workflow_rapids_json, 'r') as f:
        rapids = json.load(f)

    original_filenames = [r['asset']['originalFileName'] for r in rapids]
    prompts = [r['target'] for r in rapids]
    bboxes = []
    failed = 0
    for idx, filename in enumerate(original_filenames):
        object_class = re.match(prompt_to_class_pattern, prompts[idx])[1].lower()
        if object_class == 'airplanes':#wrong prompt fix
            object_class = 'airplane'
        try:
            bbox = get_groundtruth_for(filename, object_class, groundtruths)
            bboxes.append(bbox)
        except IndexError:
            print(f'failed to find gt for {filename}-{object_class}')
            failed += 1

    print('failed', failed)
    bboxes = np.array(bboxes)
    assert bboxes.shape == (len(rapids)-failed, 4)

    return bboxes

