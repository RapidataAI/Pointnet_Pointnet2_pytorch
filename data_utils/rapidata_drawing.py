import glob
import itertools
import json
import os
import random
import re
from collections import defaultdict
from typing import List, Tuple, Any, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

POINTS_PER_RAPID = 1024
POINTS_DIM = 3


class LineRapidDataset(Dataset):
    points_arr_file = 'points.npy'
    bboxes_arr_file = 'bboxes.npy'
    filenames_file = 'filenames.npy'
    rapid_ids_file = 'rapid_ids.npy'
    classnames_files = 'classnames.npy'

    def __init__(self, points: np.ndarray, bboxes: np.ndarray, filenames: np.ndarray, rapid_ids: np.ndarray, classnames: np.ndarray):
        # expected array: images x nPoints x POINTS_DIM
        assert len(points) == len(bboxes) == len(filenames) == len(rapid_ids) == len(classnames)

        assert len(points.shape) == 3
        assert points.shape[-1] == POINTS_DIM
        self.points = points

        assert len(bboxes.shape) == 2
        assert bboxes.shape[-1] == 4
        self.bboxes = bboxes

        assert filenames.shape == (points.shape[0],)
        self.filenames = filenames

        assert rapid_ids.shape == (points.shape[0],)
        self.rapid_ids = rapid_ids

        assert classnames.shape == (points.shape[0], )
        self.classnames = classnames

    def to_directory(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, self.points_arr_file), self.points)
        np.save(os.path.join(directory, self.bboxes_arr_file), self.bboxes)
        np.save(os.path.join(directory, self.filenames_file), self.filenames)
        np.save(os.path.join(directory, self.rapid_ids_file), self.rapid_ids)
        np.save(os.path.join(directory, self.classnames_files), self.classnames)
        return self

    @classmethod
    def from_directory(cls, directory: str, max_size=None):
        points = np.load(os.path.join(directory, cls.points_arr_file))
        labels = np.load(os.path.join(directory, cls.bboxes_arr_file))
        filenames = np.load(os.path.join(directory, cls.filenames_file))
        rapid_ids = np.load(os.path.join(directory, cls.rapid_ids_file))
        classnames = np.load(os.path.join(directory, cls.classnames_files))
        if max_size is not None:
            points = points[:max_size]
            labels = labels[:max_size]
            filenames = filenames[:max_size]
            rapid_ids = rapid_ids[:max_size]
            classnames = classnames[:max_size]
        return LineRapidDataset(points, labels, filenames, rapid_ids, classnames)

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


def get_groundtruth_for(filename: str, object_class: str, groundtruth_df: pd.DataFrame) -> List[float]:
    row = groundtruth_df[
        (groundtruth_df.file_name == filename)
        & (groundtruth_df.category_name == object_class)
        ].iloc[0]

    bbox_arr = json.loads(row.bbox)
    return bbox_arr


def get_lines_from_session(session_doc) -> List[List[Tuple[float, float]]]:
    lines = []

    for line in session_doc['result']['lines']:
        coords = []
        for point in line['points']:
            x, y = float(point['x']), float(point['y'])
            coords.append((x, y))
        lines.append(coords)

    return lines


def pad_or_clip_list(points: List[Tuple[float, float]], target_length: int, pad_value: Any):

    if len(points) < target_length:
        points = points + [pad_value] * (target_length - len(points))
    elif len(points) > target_length:
        points = random.sample(points, target_length)

    assert len(points) == target_length
    return points


def read_points_from_guesses(workflow_sessions_json: str) -> Dict[str, np.ndarray]:
    with open(workflow_sessions_json, 'r') as f:
        sessions = json.load(f)

    rapid_guesses = defaultdict(list)

    for session in sessions:
        lines = get_lines_from_session(session)

        points = list(itertools.chain(*lines))
        rapid_id = session['rapidId']['$oid']

        rapid_guesses[rapid_id] += points

    rapid_id_to_points = dict()
    for rapid_id, collected_points in rapid_guesses.items():
        points = pad_or_clip_list(collected_points, target_length=POINTS_PER_RAPID, pad_value=collected_points[0])

        points = np.array(points)
        points = points-0.5 # shift to origo

        if POINTS_DIM == 3:
            z_coords = np.random.uniform(low=-0.1, high=0.1, size=points.shape[0]).reshape(-1, 1)
            points = np.hstack((points, z_coords))
        rapid_id_to_points[rapid_id] = points

    assert all([arr.shape[-1] == POINTS_DIM for arr in rapid_id_to_points.values()]), f'points should be {POINTS_DIM} dim'
    assert all([arr.shape[-2] == POINTS_PER_RAPID for arr in rapid_id_to_points.values()]), 'Arr len should be uniform'
    assert np.min([np.min(arr) for arr in rapid_id_to_points.values()]) >= -1, 'Minimum should be at least 0.0'
    assert np.max([np.max(arr) for arr in rapid_id_to_points.values()]) <= 1, 'Maximum should be at most 1.0'
    del rapid_guesses
    return rapid_id_to_points


def read_labels_for_rapids(workflow_guesses_path: str, rapid_ids: List[str], groundtruths: pd.DataFrame,
                           prompt_to_class_pattern: str):
    with open(workflow_guesses_path, 'r') as f:
        guesses = json.load(f)

    rapid_filename_mapping = {guess['rapidId']["$oid"]: guess['originalFileName'] for guess in guesses}

    rapid_target_mapping = {guess['rapidId']["$oid"]: guess['target'] for guess in guesses}

    original_filenames = [rapid_filename_mapping[rapid_id] for rapid_id in rapid_ids]
    prompts = [rapid_target_mapping[rapid_id] for rapid_id in rapid_ids]

    bboxes = []
    classnames = []

    for idx, filename in enumerate(original_filenames):
        object_class = re.match(prompt_to_class_pattern, prompts[idx])[1].lower()
        classnames.append(object_class)
        bbox = get_groundtruth_for(filename, object_class, groundtruths)
        bboxes.append(np.array(bbox))

    assert len(bboxes) == len(rapid_ids) == len(classnames), 'sussy'
    assert all(bbox.shape == (4,) for bbox in bboxes), 'innovative bbox shape'

    return bboxes, original_filenames, classnames
