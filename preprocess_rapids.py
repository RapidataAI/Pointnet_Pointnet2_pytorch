import os

import numpy as np
import torch.utils.data
from tqdm import tqdm

from data_utils.rapidata_drawing import read_grountruths_under_folder, read_points_from_guesses, \
    read_labels_for_rapids, LineRapidDataset, POINTS_PER_RAPID, POINTS_DIM

EXPORTED_WORKFLOWS_DIR = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/exported_sessions'
PROMPT_CLASS_REGEX = 'Paint the (.*?) with your finger! Be accurate!'
OUTPUT_DIRECTORY_ROOT = os.path.join('data', 'preprocessed_datasets')
COCO_ROOT = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/coco'
ONLY_IMAGES = True

def load_dataset(split: str):
    folder = os.path.join(EXPORTED_WORKFLOWS_DIR, split)

    ground_truths = read_grountruths_under_folder(COCO_ROOT, split)

    output_directory = os.path.join(OUTPUT_DIRECTORY_ROOT, split)
    os.makedirs(output_directory, exist_ok=True)

    all_points = []
    all_labels = []
    all_filenames = []
    all_rapid_ids = []
    all_classnames = []
    n_rapids_total = 0
    for file in tqdm(os.listdir(folder)):
        workflow_guesses_path = os.path.join(folder, file)

        points_per_rapid = read_points_from_guesses(workflow_guesses_path)
        rapid_ids = list(points_per_rapid.keys())
        all_rapid_ids += rapid_ids
        n_rapids_total += len(rapid_ids)
        points = [points_per_rapid[rid] for rid in rapid_ids]
        assert all(point.shape == (POINTS_PER_RAPID, POINTS_DIM) for point in points), 'AYYAYYAY not uniform shape aaa'
        all_points += points

        if not ONLY_IMAGES:
            labels, original_filenames, classnames = read_labels_for_rapids(workflow_guesses_path, rapid_ids, ground_truths, PROMPT_CLASS_REGEX)
            all_classnames += classnames
            all_filenames += original_filenames
            assert len(labels) == len(points), 'This just doesnt feel right'
            all_labels += labels


    all_points = np.stack(all_points, axis=0)
    all_rapid_ids = np.array(all_rapid_ids)
    all_filenames = np.array(all_filenames)
    all_classnames = np.array(all_classnames)
    if not ONLY_IMAGES:
        all_labels = np.stack(all_labels, axis=0)
    else:
        all_labels = np.array([])

    assert all_points.shape == (n_rapids_total, POINTS_PER_RAPID, POINTS_DIM), f'oyyoyoy, {all_points.shape}'

    if not ONLY_IMAGES:
        assert all_labels.shape == (n_rapids_total, 4), f'4 is a better number {all_labels.shape}'
        assert all_labels.shape[0] == len(all_filenames)

    dataset = (LineRapidDataset(all_points, all_labels, all_filenames,
                               all_rapid_ids, all_classnames, only_images=ONLY_IMAGES)
               .to_directory(output_directory))

    print(f'Created dataset from {all_points.shape[0]} rapids for split {split}. Points shape {all_points.shape}, Labels shape {all_labels.shape}')

    if POINTS_DIM == 3:
        n_zeros = np.sum(np.all(all_points == [0, 0, 0], axis=-1))
    else:
        n_zeros = np.sum(np.all(all_points == [0, 0], axis=-1))

    print(f'Found {n_zeros} all-zero points out of {np.prod(all_points.shape[:-1])} points in split {split} ({round(n_zeros/np.prod(all_points.shape[:-1])*100,2)}%)')


    return dataset



def main():

    if ONLY_IMAGES:
        print('Creating inference images')
        load_dataset('inference')
        return
    val_dataset = load_dataset('val')
    train_dataset = load_dataset('train')

    print(len(train_dataset), len(val_dataset))

    val_dataset = LineRapidDataset.from_directory(os.path.join(OUTPUT_DIRECTORY_ROOT, 'inference'))
    train_dataset = LineRapidDataset.from_directory(os.path.join(OUTPUT_DIRECTORY_ROOT, 'train'))
    print(len(train_dataset), len(val_dataset))

if __name__ == '__main__':
    main()
