import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss

from data_utils.rapidata_drawing import POINTS_DIM
from models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()

        points_channel = POINTS_DIM
        bbox_output_dims = 4
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=points_channel)

        self.nn = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, bbox_output_dims),
            nn.Sigmoid()
        )


    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        #trans_feat = torch.randn((32,3,3))
        x = self.nn(x)

        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.0):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        return F.mse_loss(pred, target) + mat_diff_loss * self.mat_diff_loss_scale


        #print(pred[0].cpu().detach().numpy(), target[0].cpu().detach().numpy())
        pred = self.ensure_valid_bounding_box(pred)
        #print(pred[0].cpu().detach().numpy(), target[0].cpu().detach().numpy())
        #print('hi')
        target = self.convert_xywh_to_xyxy_bbox(target)

        loss = iou_loss(pred, target, reduction='mean')

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


    def convert_xywh_to_xyxy_bbox(self, boxes):

        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        xyxy_boxes = torch.stack([x1, y1, x2, y2], dim=1)

        return xyxy_boxes

    def ensure_valid_bounding_box(self, pred):

        x1, y1, x2, y2 = pred[:,0], pred[:,1], pred[:,2], pred[:,3]

        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        y1 = y1.unsqueeze(0)
        y2 = y2.unsqueeze(0)

        x_sorted = torch.cat([x1, x2], dim=0).sort(dim=0).values
        y_sorted = torch.cat([y1, y2], dim=0).sort(dim=0).values

        x1, x2 = x_sorted[0], x_sorted[1]
        y1, y2 = y_sorted[0], y_sorted[1]

        # Crop coordinates to the range [0, 1]
        x1 = torch.clamp(x1, 0, 1)
        x2 = torch.clamp(x2, 0, 1)
        y1 = torch.clamp(y1, 0, 1)
        y2 = torch.clamp(y2, 0, 1)

        return torch.stack([x1, y1, x2, y2], dim=1)
