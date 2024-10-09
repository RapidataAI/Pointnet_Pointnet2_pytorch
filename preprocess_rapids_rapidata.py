import os
from typing import List, Tuple

import dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm

from rapidata.services import ClickHouse, RapidataApi, COCOBoxService
from rapidata.utils import LineRapidResult, Point, RelativeCocoBox

PROMPT_CLASS_REGEX = 'Paint the (.*?) with your finger! Be accurate!'
OUTPUT_DIRECTORY_ROOT = os.path.join('data', 'preprocessed_datasets')

def dump_to_txt(points: List[Point], colors: List[Tuple[int, int , int]], file_path: str):
    assert len(points) == len(colors)

    with open(file_path, 'w+') as f:
        for point_idx, (x, y, z) in enumerate(points):
            r, g, b = colors[point_idx]
            line = f'{x} {y} {z} {r} {g} {b}\n'
            f.write(line)


def linear_interpolation(point1: Point, point2: Point, num_points=10) -> List[Point]:
    px, py, _ = point1
    prevx, prevy, _ = point2

    t_values = np.linspace(0, 1, num_points)

    interpolated_x = prevx + t_values * (px - prevx)
    interpolated_y = prevy + t_values * (py - prevy)

    interpolated_points = [Point(x,y) for x,y in zip(interpolated_x, interpolated_y)]

    return interpolated_points

def upsample_line(points: List[Point]) -> List[Point]:
    upsampled_points = []
    prev = points[0]
    for point in points[1:]:
        upsampled_points += linear_interpolation(prev, point)
    return points


def point_is_in_bbox2d(point: Point, box: RelativeCocoBox) -> bool:
    x, y, _ = point
    return box.x <= x <= box.x+box.w and box.y <= y <= box.y+box.h



def create_scenes(rapids: List[LineRapidResult], output_folder: str):
    os.makedirs(output_folder, exist_ok=True)

    for rapid in rapids:
        rapid_folder = os.path.join(output_folder, rapid.rapid.rapid_id)
        os.makedirs(rapid_folder, exist_ok=True)

        in_box_points = []
        in_box_colors = []

        out_box_points = []
        out_box_colors = []

        for line in rapid.selected_lines:
            user_score = line.user_score
            user_score_color = [user_score, user_score, user_score]
            upsampled_points = upsample_line(line.points)
            gt_box = rapid.rapid.ground_truth
            good_points = [(p.x, p.y, p.z) for p in upsampled_points if point_is_in_bbox2d(p, gt_box)]
            bad_points = [(p.x, p.y, p.z) for p in upsampled_points if not point_is_in_bbox2d(p, gt_box)]

            in_box_points += good_points
            in_box_colors += len(good_points) * [user_score_color]

            out_box_points += bad_points
            out_box_colors += len(bad_points) * [user_score_color]


        annotations_folder = os.path.join(rapid_folder, 'Annotations')
        os.makedirs(annotations_folder, exist_ok=True)
        dump_to_txt(in_box_points, in_box_colors, file_path=os.path.join(annotations_folder, 'good_1.txt'))
        dump_to_txt(out_box_points, out_box_colors, file_path=os.path.join(annotations_folder, 'bad_1.txt'))

        dump_to_txt(in_box_points+out_box_points, in_box_colors+out_box_colors,
                    file_path=os.path.join(rapid_folder, f'{rapid.rapid.rapid_id}.txt' ))

        labels = np.expand_dims(np.array([0]*len(in_box_points) + [0]*len(out_box_points)), -1)
        points = np.array(in_box_points+out_box_points)
        colors = np.array(in_box_colors+out_box_colors)
        arr = np.concatenate([points, colors, labels], axis=-1)
        np.save(os.path.join(rapid_folder, 'points.npy'), arr)


def main():
    dotenv.load_dotenv()
    os.makedirs(OUTPUT_DIRECTORY_ROOT, exist_ok=True)
    api = RapidataApi()
    gt_service = COCOBoxService()
    clickhouse = ClickHouse(api, gt_service)

    train_workflows = pd.read_csv(os.environ['BIG_TRAIN_WORKFLOWS_CSV']).workflowId.tolist()
    test_workflows = pd.read_csv(os.environ['BIG_VAL_WORKFLOWS_CSV']).workflowId.tolist()

    for split, workflows in zip(['train', 'test'], [train_workflows, test_workflows]):
        for workflow in tqdm(workflows, desc=f'Creating {split} split..'):
            rapids = clickhouse.get_line_rapid_results(workflow, PROMPT_CLASS_REGEX)
            workflow_folder = os.path.join(OUTPUT_DIRECTORY_ROOT, workflow)

            os.makedirs(workflow_folder, exist_ok=True)
            create_scenes(rapids, str(workflow_folder))


if __name__ == '__main__':
    main()
