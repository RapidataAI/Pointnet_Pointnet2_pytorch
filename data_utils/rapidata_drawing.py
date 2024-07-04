from torch.utils.data import Dataset
import numpy as np


class ModelNetDataLoader(Dataset):

    def __init__(self, points_array_path: str):
        # expected array: images x nPoints x 3
        self.points = np.load(points_array_path)
        assert len(self.points.shape) == 3
        assert self.points.shape[-1] == 3

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index]

