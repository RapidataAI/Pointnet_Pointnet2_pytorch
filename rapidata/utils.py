import abc
import dataclasses
import enum
import glob
import itertools
import json
import os
from typing import List, Optional, Tuple, Union, Any, Literal

import PIL.Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

@dataclasses.dataclass
class Point:
    x: Union[int, float]
    y: Union[int, float]
    z: Optional[Union[int, float]] = 0

    #TODO with template

    def scale_to_image(self, image: PIL.Image.Image) -> 'Point':
        return Point(self.x * image.width, self.y * image.height)

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        elif key == 2:
            return self.z
        raise IndexError


@dataclasses.dataclass
class RelativeCocoBox:
    x: float
    y: float
    w: float
    h: float
    conf: float = 1.0
    def get_absolute_coords(self) -> Tuple[float, float, float, float]:
        # Convert from (x, y, w, h) to (x1, y1, x2, y2)
        x1 = self.x
        y1 = self.y
        x2 = self.x + self.w
        y2 = self.y + self.h
        return x1, y1, x2, y2
    def iou_with(self, box: 'RelativeCocoBox') -> float:
        # Get the coordinates of both boxes
        x1, y1, x2, y2 = self.get_absolute_coords()
        x1_box, y1_box, x2_box, y2_box = box.get_absolute_coords()

        # Calculate the intersection coordinates
        inter_x1 = max(x1, x1_box)
        inter_y1 = max(y1, y1_box)
        inter_x2 = min(x2, x2_box)
        inter_y2 = min(y2, y2_box)

        # Calculate the area of intersection
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)

        inter_area = inter_w * inter_h

        # Calculate the area of both bounding boxes
        self_area = self.w * self.h
        box_area = box.w * box.h

        # Calculate IoU
        union_area = self_area + box_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0

        return iou

    @staticmethod
    def from_list(coord_ls: List[float]) -> 'RelativeCocoBox':
        assert len(coord_ls) == 4
        return RelativeCocoBox(
            coord_ls[0],
            coord_ls[1],
            coord_ls[2],
            coord_ls[3]
        )

    def to_tensor(self) -> torch.Tensor:
        return torch.Tensor([self.x, self.y, self.w, self.h])

    def to_list(self) -> List[float]:
        return [float(self.x), float(self.y), float(self.w), float(self.h)]
    def to_json(self) -> str:
        return json.dumps(self.to_list())

    def with_conf(self, conf: float) -> 'RelativeCocoBox':
        self.conf = conf
        return self


@dataclasses.dataclass
class LineSubmission:
    points: List[Point]
    user_score: float


class ImageService(abc.ABC):

    def get_image(self, image_name: str) -> PIL.Image.Image:
        raise NotImplementedError

class GroundTruthService(abc.ABC):
    def get_groundtruth(self, original_filename: str, class_name: str) -> RelativeCocoBox:
        raise NotImplementedError


class ImageRapid:
    labels_df: pd.DataFrame = None

    def __init__(self, rapid_id: str, original_filename: str, file_id: str, object_class: str,
                 image_service: ImageService, gt_service: GroundTruthService):
        self.rapid_id = rapid_id
        self.original_filename = original_filename
        self.file_id = file_id
        self.object_class = object_class

        self.image_service = image_service
        self.gt_service = gt_service
        self._image = None
        self.ground_truth = self.gt_service.get_groundtruth(self.original_filename, self.object_class)

    @property
    def image_width(self):
        if self._image is None:
            self._image = self.image_service.get_image(self.file_id)
        return self._image.width

    @property
    def image_height(self):
        if self._image is None:
            self._image = self.image_service.get_image(self.file_id)
        return self._image.height

    @property
    def image(self):
        if self._image is None:
            self._image = self.image_service.get_image(self.file_id)
        return self._image

    def _load_bbox(self) -> RelativeCocoBox:
        subset = ImageRapid.labels_df[
            (ImageRapid.labels_df.file_name == self.original_filename) &
            (ImageRapid.labels_df.category_name == self.object_class)
            ]
        assert len(subset) == 1
        return RelativeCocoBox.from_list(subset.iloc[0]['bbox'])


class LineRapidResult:
    rapid: ImageRapid
    workflow_id: str  #or the other id
    _all_lines: List[LineSubmission]
    selected_lines: List[LineSubmission]
    prediction: RelativeCocoBox

    class LineOrderByStrategy(enum.Enum):
        NEGATIVE_LENGTH = enum.auto()
        NEGATIVE_USER_SCORE = enum.auto()
        NEGATIVE_MEDIAN_IOU = enum.auto()

    def __init__(self, rapid: ImageRapid, workflow_id: str, lines: List[LineSubmission]):
        self._all_lines = [*lines]
        self.selected_lines = lines
        self.rapid = rapid
        self.workflow_id = workflow_id
        self.current_image_width = 1
        self.current_image_height = 1

    def viz(self, box: Optional[RelativeCocoBox] = None) -> PIL.Image.Image:
        raise NotImplementedError

    def register_prediction(self, coco_box: RelativeCocoBox):
        self.prediction = coco_box

    def get_iou(self) -> float:
        return self.prediction.iou_with(self.rapid.ground_truth)

    def get_xy_medians(self) -> Tuple[float, float]:
        x_med = np.median([p.x for p in self.all_points()])
        y_med = np.median([p.y for p in self.all_points()])
        return x_med, y_med

    def all_points(self) -> List[Point]:
        lines = list(map(lambda subm: subm.points, self.selected_lines))
        return list(itertools.chain(*lines))

    def scaled_points(self, image: PIL.Image.Image = None) -> List[Point]:
        if image is None:
            image = self.rapid.image

        points = self.all_points()
        return [p.scale_to_image(image) for p in points]

    def n_selected_lines(self):
        return len(self.selected_lines)

    def _get_box_around_line(self, submission: LineSubmission) -> RelativeCocoBox:
        xs = [p.x for p in submission.points]
        ys = [p.y for p in submission.points]
        mnx = min(xs)
        mny = min(ys)
        return RelativeCocoBox(mnx, mny, max(xs) - mnx, max(ys) - mny)

    def select_subset(self, keep_line_ratio: float, order_by: LineOrderByStrategy, length_outliers_last=True):
        if order_by == self.LineOrderByStrategy.NEGATIVE_USER_SCORE:
            ordered_lines = sorted(self._all_lines, key=lambda subm: -1 * subm.user_score)
        elif order_by == self.LineOrderByStrategy.NEGATIVE_LENGTH:
            ordered_lines = sorted(self._all_lines, key=lambda subm: -1* len(subm.points))
        elif order_by == self.LineOrderByStrategy.NEGATIVE_MEDIAN_IOU:
            line_boxes = list(map(self._get_box_around_line, self._all_lines))
            median_ious = []

            for line_idx, line in enumerate(self._all_lines):
                ious = []
                for other_line_idx, other_line in enumerate(self._all_lines):
                    line_bbox = line_boxes[line_idx]
                    other_line_bbox = line_boxes[other_line_idx]

                    iou = line_bbox.iou_with(other_line_bbox)
                    ious.append(iou)
                median_ious.append(-1 * np.median(ious))

            idxs_sorted = np.argsort(median_ious)
            ordered_lines = [self._all_lines[idx] for idx in idxs_sorted]
        else:
            raise NotImplementedError(f'Unknown Strategy {order_by}')

        cut = int(keep_line_ratio * len(self._all_lines))
        if length_outliers_last:
            suspicious_lines_indices = [idx for idx in range(len(ordered_lines))
                                        if len(ordered_lines[idx].points) < 5 or len(ordered_lines[idx].points) > 100]
            removed = []
            for idx in suspicious_lines_indices[::-1]:
                removed.append(ordered_lines[idx])
                del ordered_lines[idx]
            for line in removed[::-1]:
                ordered_lines.append(line)

        self.selected_lines = ordered_lines[:cut]

    def select_random_sample(self, keep_line_ratio: float):
        raise NotImplementedError


class LineRapidResultBuilder:

    def __init__(self, image_service: ImageService, gt_service: GroundTruthService):
        self.image_service = image_service
        self.gt_service = gt_service
        self.lines = []
        self.workflow_id: str = None
        self.rapid_id: str = None
        self.rapid_class_name: str = None
        self.rapid_filename: str = None
        self.rapid_file_id: str = None

    def with_line(self, line: LineSubmission):
        self.lines.append(line)

    def with_workflow_id(self, workflow_id: str):
        self.workflow_id = workflow_id

    def with_rapid_id(self, rapid_id: str):
        self.rapid_id = rapid_id

    def with_rapid_class_name(self, class_name: str):
        self.rapid_class_name = class_name

    def with_file_id(self, filename: str):
        self.rapid_file_id = filename

    def with_original_filename(self, filename: str):
        self.rapid_filename = filename

    def build(self) -> LineRapidResult:
        assert len(self.lines) > 0
        assert self.workflow_id is not None
        assert self.rapid_id is not None
        assert self.rapid_class_name is not None
        assert self.rapid_filename is not None
        assert self.rapid_file_id is not None

        return LineRapidResult(
            lines=self.lines,
            workflow_id=self.workflow_id,
            rapid=ImageRapid(
                original_filename=self.rapid_filename,
                file_id=self.rapid_file_id,
                rapid_id=self.rapid_id,
                object_class=self.rapid_class_name,
                image_service=self.image_service,
                gt_service=self.gt_service
            ),
        )


class PointsToImageProcessor(abc.ABC):
    def process(self, results: LineRapidResult) -> np.ndarray:
        raise NotImplementedError

    def process_batch(self, results: List[LineRapidResult]) -> List[np.ndarray]:
        return [self.process(r) for r in results]


class PointsToGreyScaleImageProcessor(PointsToImageProcessor):
    def process(self, result: LineRapidResult) -> np.ndarray:
        raise NotImplementedError


class PointsToRGBImage(PointsToImageProcessor):
    class RGBModes(enum.Enum):
        LowMediumHighUserScores = enum.auto()
        LinesPointsValues = enum.auto()

    def __init__(self, mode: RGBModes, target_image_size: Tuple[int, int]):
        assert mode in self.RGBModes
        self.mode = mode
        self.target_image_size = target_image_size

    def draw_bresenham_line(self, p1: Tuple[int, int], p2: Tuple[int, int], value: Any, image: np.ndarray,
                            agg: Literal["addition", "overwrite", "max"] = 'overwrite'):
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

    def process(self, result: LineRapidResult) -> np.ndarray:
        IMAGE_SIZE = self.target_image_size

        first_channel = np.zeros(IMAGE_SIZE)
        second_channel = np.zeros(IMAGE_SIZE)
        third_channel = np.zeros(IMAGE_SIZE)

        for line_idx, line in enumerate(result.selected_lines):
            prev_x, prev_y = line.points[0]
            prev_x, prev_y = min(0.9999, prev_x), min(0.9999, prev_y)
            prev_x, prev_y = int(prev_x * IMAGE_SIZE[0]), int(prev_y * IMAGE_SIZE[1])

            for point_idx, (x, y) in enumerate(line.points):

                x, y = min(0.9999, x), min(0.9999, y)
                x, y = int(x * IMAGE_SIZE[0]), int(y * IMAGE_SIZE[1])

                if self.mode == self.RGBModes.LinesPointsValues:
                    self.draw_bresenham_line((prev_x, prev_y), (x, y), 1.0, first_channel, agg='addition')
                    self.draw_bresenham_line((prev_x, prev_y), (x, y), line_idx, second_channel, agg='overwrite')
                    self.draw_bresenham_line((prev_x, prev_y), (x, y), point_idx, third_channel, agg='overwrite')

                elif self.mode == self.RGBModes.LowMediumHighUserScores:
                    line_user_score = line.user_score
                    if line_user_score < 0.5:
                        self.draw_bresenham_line((prev_x, prev_y), (x, y), line_user_score, first_channel, agg='max')
                    elif 0.5 < line_user_score < 0.75:
                        self.draw_bresenham_line((prev_x, prev_y), (x, y), line_user_score, second_channel, agg='max')
                    else:
                        self.draw_bresenham_line((prev_x, prev_y), (x, y), line_user_score, third_channel, agg='max')

                prev_x, prev_y = x, y

        first_channel = (first_channel / np.max(first_channel)) * 255
        second_channel = (second_channel / np.max(third_channel)) * 255
        third_channel = (third_channel / np.max(second_channel)) * 255

        return np.stack([third_channel, second_channel, first_channel], axis=-1)


class TopNLinesProcessor(PointsToImageProcessor):

    def __init__(self, n, order_by: LineRapidResult.LineOrderByStrategy, image_size: Tuple[int, int]):
        self.n = n
        self.image_size = image_size
        self.order_by = order_by

    def draw_bresenham_line(self, p1: Tuple[int, int], p2: Tuple[int, int], value: Any, image: np.ndarray,
                            agg: Literal["addition", "overwrite", "max"] = 'overwrite'):
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

    def scale_point(self, x, y) -> Tuple[int, int]:
        x, y = min(x, 0.999999), min(y, 0.999999)

        return int(x * self.image_size[0]), int(y * self.image_size[0])

    def process(self, result: LineRapidResult) -> np.ndarray:

        user_lines = np.zeros(shape=(self.n, *self.image_size))

        ratio = self.n / len(result._all_lines)

        result.select_subset(
            keep_line_ratio=ratio,
            order_by=self.order_by,
            length_outliers_last=True
        )

        for line_idx, line in enumerate(result.selected_lines):
            prev_x, prev_y = line.points[0]
            prev_x, prev_y = self.scale_point(prev_x, prev_y)
            for x, y in line.points:
                x, y = self.scale_point(x, y)
                self.draw_bresenham_line((prev_x, prev_y), (x, y),
                                         value=line.user_score, agg='max', image=user_lines[line_idx])

                prev_x, prev_y = x, y

        return user_lines


@dataclasses.dataclass
class ZoomedImage:
    original_image_filename: str
    new_image: PIL.Image.Image
    relative_zoom_box: RelativeCocoBox

    def get_new_filename(self) -> str:
        filename, extension = self.original_image_filename.split('.')
        return filename+'_zoomed.'+extension


    def get_new_bounding_box(self, original_bounding_box: RelativeCocoBox) -> RelativeCocoBox:


        x1, y1, w, h = original_bounding_box.to_list()
        zx1, zy1, zw, zh = self.relative_zoom_box.to_list()

        new_x1 = (x1 - zx1) / zw
        new_y1 = (y1 - zy1) / zh
        new_w = w / zw
        new_h = h / zh

        return RelativeCocoBox(new_x1, new_y1, new_w, new_h)

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.new_image.save(os.path.join(folder, self.get_new_filename()))



class RapidDataSet(Dataset):


    def __init__(self, X: torch.Tensor, Y: torch.Tensor, rapid_ids: List):
        self.X = X
        self.Y = Y
        self.rapid_ids = rapid_ids
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)

def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch