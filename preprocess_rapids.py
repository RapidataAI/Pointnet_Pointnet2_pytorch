import os

import numpy as np
import torch.utils.data

from data_utils.rapidata_drawing import read_grountruths_under_folder, get_groundtruth_for, read_points_from_workflow, \
    read_labels_from_workflow, LineRapidDataLoader

EXPORTED_WORKFLOWS_DIR = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/exported_workflows'
PROMPT_CLASS_REGEX = 'Paint the (.*?) with your finger! Be accurate!'
OUTPUT_DIRECTORY_ROOT = os.path.join('data', 'preprocessed_datasets')
COCO_ROOT = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/coco'

def load_dataset(split: str):
    folder = os.path.join(EXPORTED_WORKFLOWS_DIR, split)
    ground_truths = read_grountruths_under_folder(COCO_ROOT, split)

    output_directory = os.path.join(OUTPUT_DIRECTORY_ROOT, split)
    os.makedirs(output_directory, exist_ok=True)
    datasets = []
    for file in os.listdir(folder):
        abs_path = os.path.join(folder, file)
        points = read_points_from_workflow(abs_path)
        labels = read_labels_from_workflow(abs_path, ground_truths, PROMPT_CLASS_REGEX)
        dataset_name = file.replace('.', '')
        dataset = LineRapidDataLoader(points, labels).to_directory(
            os.path.join(
                output_directory, dataset_name
                )
        )
        datasets.append(dataset)

    return torch.utils.data.ConcatDataset(datasets)



def main():
    val_dataset = load_dataset('val')
    train_dataset = load_dataset('train')

    print(len(train_dataset), len(val_dataset))

    val_dataset = LineRapidDataLoader.concat_dataset_from_root(os.path.join(OUTPUT_DIRECTORY_ROOT, 'val'))
    train_dataset = LineRapidDataLoader.concat_dataset_from_root(os.path.join(OUTPUT_DIRECTORY_ROOT, 'train'))
    print(len(train_dataset), len(val_dataset))

if __name__ == '__main__':
    main()
