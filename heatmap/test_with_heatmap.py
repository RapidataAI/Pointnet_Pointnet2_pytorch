import dataclasses
import glob
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
from tqdm import tqdm
import os
import math
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

yolo_labels_root = '/home/ubuntu/Pointnet_Pointnet2_pytorch/conv_nets/datasets/yolo_rapids_heatmap_image_64x64/labels/inference'
output_folder = 'results'
os.makedirs(output_folder, exist_ok=True)


@dataclasses.dataclass
class WorkflowResult:
    workflow_id: str
    points_per_rapid: Dict
    point_weights_per_rapid: Dict
    rapid_original_filenames: Dict
    ious: Dict = dataclasses.field(default_factory=dict)
    coco_preds: Dict = dataclasses.field(default_factory=dict)
    image_sizes: Dict = dataclasses.field(default_factory=dict)




def get_lines_from_guess(guess: Dict) -> List[List[Tuple[float, float]]]:
    all_lines = []
    for line in guess['result']['lines']:
        points = []
        for point in line['points']:
            x, y = float(point['x']), float(point['y'])
            points.append((x, y))
        all_lines.append(points)
    return all_lines



def interpolate_points(p1: Tuple[float,float], p2: Tuple[float,float], n_points: int):
    points = []
    for i in range(n_points):
        x = p1[0] + (p2[0]-p1[0]) * (i/n_points)
        y = p1[1] + (p2[1]-p1[1]) * (i/n_points)
        points.append((x,y))

    return points

def upsample_line(line: List[Tuple[float, float]]):

    prev_x, prev_y = line[0][0]*100, line[0][1]*100
    points = []
    for point in line:
        x, y = point
        x, y = x*100, y*100
        distance = math.sqrt((prev_x - x) ** 2 + (prev_y - y) ** 2)
        distance_factor = 1
        number_of_points = int(distance*distance_factor)

        new_points = interpolate_points((prev_x, prev_y), (x,y), number_of_points)

        points += [(x/100, y/100) for x,y in new_points]

        prev_x, prev_y = x, y

    weights = [len(line) / 52] *len(points)
    return points, weights


def get_original_filename_from_guess(guess: Dict):
    return guess['originalFileName']


def get_workflow_id_from_guess(guess: Dict):
    return guess["workflowId"]["$oid"]

def get_rapid_id_from_guess(guess: Dict):
    return guess['rapidId']['$oid']

def get_workflow_result(workflow_guesses: List[Dict]) -> WorkflowResult:

    rapids_points = defaultdict(list)
    rapid_point_weights = defaultdict(list)
    filenames = dict()
    workflow_id = get_workflow_id_from_guess(workflow_guesses[0])

    for guess in workflow_guesses:
        lines = get_lines_from_guess(guess)

        rapid_id = get_rapid_id_from_guess(guess)

        filenames[rapid_id] = get_original_filename_from_guess(guess)

        for line in lines:
            line_points, line_weights = upsample_line(line)
            rapids_points[rapid_id] += line_points
            rapid_point_weights[rapid_id]  += line_weights

    result = WorkflowResult(
        workflow_id=workflow_id,
        points_per_rapid=rapids_points,
        point_weights_per_rapid=rapid_point_weights,
        rapid_original_filenames=filenames
    )

    calc_preds_and_image(result)
    calc_ious(result)
    return result


def read_image(original_filename: str):
    image_root_folder = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/coco/result_instances_val2017'
    file = glob.glob(f'{image_root_folder}/**/{original_filename}', recursive=True)[0]
    image = Image.open(file)
    return image


def calc_preds_and_image(result: WorkflowResult):
    fig, ax = plt.subplots()
    heatmap_levels=2
    for rapid_id in list(result.points_per_rapid.keys()):
        points = result.points_per_rapid[rapid_id]
        image = read_image(result.rapid_original_filenames[rapid_id])
        result.image_sizes[rapid_id] = (image.width, image.height)

        x_points = [p[0] * image.width for p in points]
        y_points = [p[1] * image.height for p in points]

        x_grid = np.linspace(0, image.width, 100)
        y_grid = np.linspace(0, image.height, 100)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

        positions = np.vstack([x_mesh.ravel(), y_mesh.ravel()])

        kernel = gaussian_kde(np.vstack([x_points, y_points]), bw_method=0.1)
        z = np.reshape(kernel(positions).T, x_mesh.shape)


        contour = ax.contourf(x_mesh, y_mesh, z, levels=heatmap_levels, cmap='turbo', alpha=0.6)
        contour = sorted(contour.collections,
                                 key=lambda x: x.get_paths()[0].vertices[:, 0].max() if x.get_paths() else 0,
                                 reverse=True)[heatmap_levels-1]

        global_x_min, global_x_max, global_y_min, global_y_max = image.width, 0, image.height, 0
        for path in contour.get_paths():
            v = path.vertices
            x_min, y_min = v.min(axis=0)
            x_max, y_max = v.max(axis=0)
            global_x_min = min(x_min, global_x_min)
            global_x_max = max(x_max, global_x_min)
            global_y_min = min(y_min, global_y_min)
            global_y_max = max(y_max, global_y_min)

        width, height = global_x_max - global_x_min, global_y_max - global_y_min
        result.coco_preds[rapid_id] = [global_x_min, global_y_min, width, height]


def viz_workflow(result: WorkflowResult):
    heatmap_levels = 2
    fig, axes = plt.subplots(nrows=3, ncols=3)

    rapids = list(result.points_per_rapid.keys())

    for ax_idx, ax in enumerate(axes.flat):
        if len(rapids) <= ax_idx:
            break
        rapid_id = rapids[ax_idx]
        points = result.points_per_rapid[rapid_id]
        weights = result.point_weights_per_rapid[rapid_id]
        image = read_image(result.rapid_original_filenames[rapid_id])


        x_points = [p[0]*image.width for p in points]
        y_points = [p[1]*image.height for p in points]

        # Create a grid of points
        x_grid = np.linspace(0, image.width, 100)
        y_grid = np.linspace(0, image.height, 100)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

        positions = np.vstack([x_mesh.ravel(), y_mesh.ravel()])

        kernel = gaussian_kde(np.vstack([x_points, y_points]), bw_method=0.1)
        z = np.reshape(kernel(positions).T, x_mesh.shape)

        ax.imshow(image, extent=(0, image.width, image.height, 0), aspect='auto')

        ax.contourf(x_mesh, y_mesh, z, levels=heatmap_levels, cmap='turbo', alpha=0.6)
        x_min, y_min, width, height = result.coco_preds[rapid_id]
        rect = Rectangle((x_min, y_min), width, height, fill=False, color="red", linewidth=2)
        ax.add_patch(rect)

        gt_xmin, gt_ymin, gt_width, gt_height = get_coco_gt(rapid_id, result.image_sizes[rapid_id])
        gt_rect = Rectangle((gt_xmin, gt_ymin), gt_width, gt_height, fill=False, color="green", linewidth=2)
        ax.add_patch(gt_rect)

        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.imshow(z, origin='lower', aspect='auto', cmap='turbo')
        ax.set_title(f'iou {round(result.ious[rapid_id], 2)}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder,f'{result.workflow_id}_samples.png'))



def get_coco_gt(rapid_id: str, image_size: Tuple[int, int]) -> Tuple:
    with open(os.path.join(yolo_labels_root, f'{rapid_id}.txt'), 'r') as f:
        line = f.readline()

    yolo_gt = list(map(float, line.split(' ')))[1:]
    assert len(yolo_gt) == 4

    cx, cy, w, h = yolo_gt
    img_width, img_height = image_size
    x = (cx * img_width) - (w * img_width / 2)
    y = (cy * img_height) - (h * img_height / 2)
    w = w * img_width
    h = h * img_height
    return (int(x), int(y), int(w), int(h))




# Calculate IoU function
def calculate_iou_coco(box1, box2):
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

def calc_ious(result: WorkflowResult):

    for rapid_id in result.points_per_rapid.keys():
        coco_gt = get_coco_gt(rapid_id, result.image_sizes[rapid_id])
        iou = calculate_iou_coco(coco_gt, result.coco_preds[rapid_id])

        result.ious[rapid_id] = iou


def main():
    exported_workflows_dir =  '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/exported_sessions/val'
    ious = []
    for workflow_guesses_file in tqdm(os.listdir(exported_workflows_dir)):
        with open(os.path.join(exported_workflows_dir, workflow_guesses_file), 'r') as f:
            workflow_guesses = json.load(f)

        workflow_result = get_workflow_result(workflow_guesses)
        ious += list(workflow_result.ious.values())
        viz_workflow(workflow_result)

    with open(os.path.join(output_folder ,'ious.json'), 'w') as f:
        json.dump(ious, f)



if __name__ == '__main__':
    main()