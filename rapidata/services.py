import abc
import json
import os
import re
import sqlite3
from collections import defaultdict
from io import BytesIO

from typing import List

import PIL
import clickhouse_connect
import pandas as pd
import requests
from PIL.Image import Image

from .utils import LineRapidResult, ZoomedImage, LineSubmission, Point, \
    LineRapidResultBuilder, ImageService, RelativeCocoBox, GroundTruthService


class TransformationDB:
    pass


class RapidataApi(ImageService):

    def __init__(self):
        jwt_token = os.environ['RAPIDATA_HTTPS_TOKEN']
        self.headers = {
            "Authorization": f"Bearer {jwt_token}"
        }

        self.image_cache_folder = os.environ['LOCAL_IMAGE_CACHE_FOLDER']
        os.makedirs(self.image_cache_folder, exist_ok=True)

    def linerapid_order_from_zoomed_images(self, images: List[ZoomedImage]) -> str:
        #return order id
        raise NotImplementedError

    def get_image(self, image_name: str) -> Image:
        if os.path.exists(im_path := os.path.join(self.image_cache_folder, image_name)):
            return PIL.Image.open(im_path)

        url = "https://assets.rapidata.ai/{file_name}".format(file_name=image_name)
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        image = PIL.Image.open(BytesIO(response.content))

        image.save(os.path.join(self.image_cache_folder, image_name))

        return image


class WorkflowResultReader(abc.ABC):
    def get_line_rapid_results(self, workflow_id: str, prompt2class_regex: str) -> List[
        LineRapidResult]:
        raise NotImplementedError


class WorkflowJsonExportReader(WorkflowResultReader):

    def __init__(self, folder: str):
        self.folder = folder

    def create_line_submissions(self, lines: List, user_score: float) -> List[LineSubmission]:
        submissions = []

        for line in lines:
            submissions.append(
                LineSubmission(
                    user_score=user_score,
                    points=[Point(float(p['x']), float(p['y'])) for p in line['points']],
                )
            )
        return submissions

    def get_line_rapid_results(self, workflow_id: str, prompt2class_regex: str, api: RapidataApi) -> List[
        LineRapidResult]:
        with open(os.path.join(self.folder, workflow_id+'.json'), "r") as f:
            content = json.load(f)

        rapid_builders = defaultdict(lambda: LineRapidResultBuilder(api))
        for guess in content:
            workflow_id, rapid_id = guess["targetGroupId"]["$oid"], guess["rapidId"]["$oid"]
            user_score, lines = guess['userScore'], guess['result']['lines']
            file_id, original_filename, prompt = guess['fileName'], guess['originalFileName'], guess['target']

            class_name = re.match(prompt2class_regex, prompt)[1].strip().lower()

            line_result_builder = rapid_builders[rapid_id]

            line_submissions = self.create_line_submissions(lines, user_score)
            for line in line_submissions:
                line_result_builder.with_line(line)

            line_result_builder.with_workflow_id(workflow_id)
            line_result_builder.with_rapid_id(rapid_id)
            line_result_builder.with_original_filename(original_filename)
            line_result_builder.with_file_id(file_id)
            line_result_builder.with_rapid_class_name(class_name)

        del content
        results = [builder.build() for builder in rapid_builders.values()]

        return results

class ClickHouse(WorkflowResultReader):
    WORKFLOW_QUERY = \
        """
    SELECT "target_group_id", "rapid_id", "user_score", "lines", "filename", "original_filename", "line_target"
    from
        (
            SELECT "target_group_id", "rapid_id", "user_score", "lines"
            FROM rapidata_dbt.user_guesses
            WHERE target_group_id = '{target_group_id}'
        ) guesses
    INNER JOIN
        (
            SELECT simpleJSONExtractString(asset, 'fileName') as filename,
             simpleJSONExtractString(asset, 'originalFileName') as original_filename,
                   line_target, id
            FROM rapidata_dbt.rapids
        ) rapids
    on guesses.rapid_id = rapids.id
    """

    def __init__(self, api: RapidataApi, gt_service: GroundTruthService):
        self.client = clickhouse_connect.get_client(
            host=os.environ['CLICKHOUSE_DB_HOST'],
            username=os.environ['CLICKHOUSE_DB_USER'],
            password=os.environ['CLICKHOUSE_DB_PASSWORD'],
            connect_timeout=5,
            secure=True
        )

        self.api = api
        self.gt_service = gt_service

    def create_line_submissions(self, lines: List, user_score: float) -> List[LineSubmission]:
        submissions = []

        for line in lines:
            submissions.append(
                LineSubmission(
                    user_score=user_score,
                    points=[Point(p['x'], p['y']) for p in line[1]],
                )
            )
        return submissions

    def get_line_rapid_results(self, target_group_id: str, prompt2class_regex: str) -> List[
        LineRapidResult]:
        query = self.WORKFLOW_QUERY.format(target_group_id=target_group_id)
        rapid_builders = defaultdict(lambda: LineRapidResultBuilder(self.api, self.gt_service))

        with self.client.query_row_block_stream(query) as stream:
            for block in stream:
                for guess in block:
                    target_group_id, rapid_id = guess[0], guess[1]
                    user_score, lines = guess[2], guess[3]
                    file_id, original_filename, prompt = guess[4], guess[5], guess[6]

                    class_name = re.match(prompt2class_regex, prompt)[1].strip().lower()
                    if class_name in ['planes', 'plane']:
                        class_name = 'airplane'

                    line_result_builder = rapid_builders[rapid_id]
                    line_submissions = self.create_line_submissions(lines, user_score)
                    for line in line_submissions:
                        line_result_builder.with_line(line)

                    line_result_builder.with_workflow_id(target_group_id)
                    line_result_builder.with_rapid_id(rapid_id)
                    line_result_builder.with_original_filename(original_filename)
                    line_result_builder.with_file_id(file_id)
                    line_result_builder.with_rapid_class_name(class_name)

        results = [builder.build() for builder in rapid_builders.values()]
        return results


class COCOBoxService(GroundTruthService):

    labels_df: pd.DataFrame = None
    DB_PATH = 'groundtruths.sqlite'

    def __init__(self):
        self._create_table_if_not_exists()
        if COCOBoxService.labels_df is None:
            COCOBoxService.labels_csv = os.environ.get('COCO_GROUND_TRUTH_FILE')
            COCOBoxService.labels_df = pd.read_csv(COCOBoxService.labels_csv)
            COCOBoxService.labels_df['bbox'] = self.labels_df['bbox'].apply(json.loads)

    def _create_table_if_not_exists(self):
        with sqlite3.connect(self.DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                 CREATE TABLE IF NOT EXISTS groundtruthes (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT NOT NULL,
                     class_name TEXT NOT NULL,
                     box TEXT NOT NULL
                 )
             ''')
            conn.commit()
    def get_groundtruth(self, original_filename: str, class_name: str) -> RelativeCocoBox:
        subset = COCOBoxService.labels_df[
            (COCOBoxService.labels_df.file_name == original_filename) &
            (COCOBoxService.labels_df.category_name == class_name)
            ]
        if len(subset) != 0:
            assert len(subset) == 1
            return RelativeCocoBox.from_list(subset.iloc[0]['bbox'])

        with sqlite3.connect(self.DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                    SELECT box FROM groundtruthes
                    WHERE filename = ? AND class_name = ?
                ''', (original_filename, class_name))

            result = cursor.fetchone()
            if not result:
                raise ValueError(f'Did not find groundtruth for {original_filename} and {class_name}')

            box_json = result[0]
            box_data = json.loads(box_json)
            box = RelativeCocoBox(
                x=box_data['x'],
                y=box_data['y'],
                w=box_data['w'],
                h=box_data['h']
            )
            return box

    def insert_groundtruth(self, filename: str, class_name: str, box: RelativeCocoBox):
        box_json = box.to_json()
        with sqlite3.connect(self.DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO groundtruthes (filename, class_name, box)
                VALUES (?, ?, ?)
            ''', (filename, class_name, box_json))
            conn.commit()
