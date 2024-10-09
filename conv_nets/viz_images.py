import os
import random
from pathlib import Path
from typing import List

import matplotlib
import glob

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

dataset_path = '/home/ubuntu/Pointnet_Pointnet2_pytorch/conv_nets/datasets/yolo_rapids_user_score_lines_128x128_us0.0'
coco_path = '/home/ubuntu/Pointnet_Pointnet2_pytorch/data/coco'


def read_metadatas(folder: str):
    dfs = []
    for file in os.listdir(folder):
        df = pd.read_csv(os.path.join(folder, file))
        df['workflow_id'] = Path(file).stem
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    return df


def normalize_point(x, y, original_width, original_height, target_size=640):
    scale_x = target_size / original_width
    scale_y = target_size / original_height
    x_normalized = x * scale_x
    y_normalized = y * scale_y

    x_normalized = min(max(x_normalized, 0), target_size)
    y_normalized = min(max(y_normalized, 0), target_size)

    return x_normalized, y_normalized


def main():
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.tight_layout(pad=3.0)

    images_dir = os.listdir(os.path.join(dataset_path, 'images', 'inference'))

    metadata_path = os.path.join(dataset_path, 'metadata')
    metadata_df = read_metadatas(metadata_path)
    sampled_images = random.sample(images_dir, 9)

    sampled_images[0] = '668be69c690d2839547f8e51.png'
    sampled_images[1] = '668bec6bcc23b57bc10a4d50.png'
    for ax, image_name in zip(axs.flatten(), sampled_images):

        image_path = os.path.join(dataset_path, 'images', 'inference', image_name)

        generated_image = Image.open(image_path)
        generated_image = generated_image.convert("RGB")
        generated_image_array = np.array(generated_image)

        label_filename = Path(image_name).stem + '.txt'
        label_path = os.path.join(dataset_path, 'labels', 'inference', label_filename)

        metadata = metadata_df[(metadata_df.rapid_ids == Path(image_name).stem)]
        assert len(metadata) == 1
        class_name = metadata.class_names.iloc[0]

        ax.imshow(generated_image)

        with open(label_path, 'r') as file:
            for line in file.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())

                h, w, _ = generated_image_array.shape
                x_center *= w
                y_center *= h
                width *= w
                height *= h
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)

                rect = plt.Rectangle((x_min, y_min), width, height, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        original_filename = metadata.filenames.iloc[0]
        original_image_path = glob.glob(f'{coco_path}/**/{original_filename}', recursive=True)[0]

        # Read the original image
        original_image = Image.open(original_image_path)
        original_image = original_image.convert("RGB")

        # Resize original image to match the generated image size
        original_image = original_image.resize(generated_image.size)

        # Create the final image by combining original and generated images
        original_image_array = np.array(original_image)
        final_image_array = np.where(generated_image_array == [0, 0, 0], original_image_array, generated_image_array)

        ax.imshow(final_image_array)
        ax.set_title(f'{class_name}-{image_name}')
        ax.axis('off')

    plt.savefig('yolo_grid.png')
    plt.close()


if __name__ == '__main__':
    main()
