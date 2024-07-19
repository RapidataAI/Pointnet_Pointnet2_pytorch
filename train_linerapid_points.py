"""
Author: Benny
Date: Nov 2019
"""
import json
import os
import sys
from collections import defaultdict

import torch
import numpy as np

import datetime
import logging

from torch.utils.data import DataLoader

import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.rapidata_drawing import LineRapidDataset, POINTS_DIM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls_bbox', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, criterion):
    losses = []
    for batch_id, (points, target) in tqdm(enumerate(loader, 0), total=len(loader), smoothing=0.9):

        points = points.data.numpy()
        points = provider.random_point_dropout(points)
        points[:, :, 0:POINTS_DIM] = provider.random_scale_point_cloud(points[:, :, 0:POINTS_DIM])
        points[:, :, 0:POINTS_DIM] = provider.shift_point_cloud(points[:, :, 0:POINTS_DIM])
        points = torch.Tensor(points)
        points = points.transpose(2, 1)

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        pred, trans_feat = model(points)
        pred = pred.float()
        target = target.float()

        loss = criterion(pred, target, trans_feat)


        losses.append(loss.item())

    return losses


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/preprocessed_datasets/'
    train_dataset = LineRapidDataset.from_directory(os.path.join(data_path, 'train'), max_size=None)
    val_dataset = LineRapidDataset.from_directory(os.path.join(data_path, 'val'), max_size=None)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    print(f'Loaded datasets. Train length: {len(train_dataset)}, val length: {len(val_dataset)}')

    '''MODEL LOADING'''
    print(args.model, os.getcwd())
    module = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_linerapid_points.py', str(exp_dir))

    model = module.get_model()
    criterion = module.get_loss(mat_diff_loss_scale=0.001)
    model.apply(inplace_relu)

    if not args.use_cpu:
        model = model.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_loss = np.inf

    '''TRANING'''
    logger.info('Start training...')
    train_logs = defaultdict(list)
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        losses = []
        model = model.train()
        for batch_id, (points, target) in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:POINTS_DIM] = provider.random_scale_point_cloud(points[:, :, 0:POINTS_DIM])
            points[:, :, 0:POINTS_DIM] = provider.shift_point_cloud(points[:, :, 0:POINTS_DIM])

            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = model(points)

            pred = pred.float()
            target = target.float()
            loss = criterion(pred, target, trans_feat)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            global_step += 1


        train_loss_mean = np.mean(losses)
        train_logs['train_loss'].append(train_loss_mean)
        log_string('Train Instance Loss: %f' % train_loss_mean)
        with torch.no_grad():
            batch_losses = test(model.eval(), val_dataloader, criterion)
            mean_loss = np.mean(batch_losses)
            train_logs['val_losses'].append(mean_loss)

            if (best_loss > mean_loss):
                best_loss = mean_loss
                best_epoch = epoch + 1
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_loss': mean_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

            log_string(f'Current val Loss: {float(mean_loss)}. (Best : {best_loss})')

            global_epoch += 1

            current_lr = optimizer.param_groups[0]['lr']
            train_logs['lr'].append(current_lr)
            scheduler.step()

    for i in zip(pred, target):
        print(i)
    savepath = os.path.join(log_dir, 'logs.json')
    with open(savepath, 'w') as f:
        json.dump(train_logs, f)
    logger.info('End of training...')




if __name__ == '__main__':
    args = parse_args()
    main(args)
