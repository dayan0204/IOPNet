import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):

        if pred.shape != target.shape:
            raise ValueError("Prediction and target shapes must match")
        valid_mask = (target>0).detach()
        if not valid_mask.any():
            return pred.sum() * 0.0
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):

        if pred.shape != target.shape:
            raise ValueError("Prediction and target shapes must match")
        valid_mask = (target>0).detach()
        if not valid_mask.any():
            return pred.sum() * 0.0
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss

class DistributionLoss(nn.Module):
    def __init__(self, num_intervals=7, interval_length=1.0):
        super(DistributionLoss, self).__init__()
        self.num_intervals = num_intervals
        self.interval_length = interval_length

    def forward(self, pred, target, range_min=1, range_max=8, bin_width=1):

        dist_matrix = torch.cdist(pred, target, p=1)
        loss = torch.mean(torch.min(dist_matrix, dim=1).values)
        


        return loss
