"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from flask.cli import load_dotenv

from data_utils.S3DISDataLoader import ScannetDatasetWholeScene, LineRapidDataset
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['IN!!!', 'VeryOut:(']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=False, default='2024-08-27_08-44', help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    load_dotenv()
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 2
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    test_workflows = pd.read_csv(os.environ['BIG_VAL_WORKFLOWS_CSV']).workflowId.tolist()

    TEST_DATASET = LineRapidDataset('data/preprocessed_datasets', test_workflows)


    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    os.makedirs('eval_viz', exist_ok=True)
    test_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    with torch.no_grad():


        rapid_counter = 0
        for i, (data_points, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
            og_target = target.cpu().numpy()
            preds = np.zeros_like(target)
            for _ in range(args.num_votes):
                points = data_points.data.numpy()
                points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                seg_pred, trans_feat = classifier(points)
                pred_choice = seg_pred.cpu().data.max(-1)[1].numpy()

                preds += pred_choice

            preds /= args.num_votes

            points = data_points.data.numpy()
            for sample_idx  in range(len(preds)):
                fig, axes = plt.subplots(nrows=2)
                x = points[sample_idx, :, 0]
                y = points[sample_idx, :, 1]
                rapid_id = TEST_DATASET.all_rapids[rapid_counter]
                axes[0].scatter(x, y, c=og_target[sample_idx], s=1)
                axes[0].set_title('GT')
                axes[1].set_title('PRED')
                axes[1].set_box_aspect(1)
                axes[0].set_box_aspect(1)
                axes[1].scatter(x, y, c=np.round(preds[sample_idx]), s=1)
                plt.title(rapid_id)
                plt.tight_layout()
                plt.savefig(os.path.join('eval_viz', f'point_preds_{rapid_id}.png'))
                plt.close()
                rapid_counter += 1


        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
