import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)


class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        logits = logits.double()
        targets = targets.double()
        # print(f"logits min: {logits.min()}, max: {logits.max()}")
        # print(f"targets min: {targets.min()}, max: {targets.max()}")
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union_logits = torch.sum(logits[:, class_index, :, :, :])
            union_targets = torch.sum(targets[:, class_index, :, :, :])
            union = union_logits + union_targets
            # print(f"Class {class_index}: inter={inter.item()}, union_logits={union_logits.item()}, union_targets={union_targets.item()}, union={union.item()}")
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.asarray(dices)


class TverskyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        dice = 0.

        for i in range(pred.size(1)):
            dice += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) +
                        0.3 * (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * (
                                    (1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2)

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, edge_output, edge_targets):
        loss = self.criterion(edge_output, edge_targets)
        return loss


class IoUAverage(object):
    """Computes and stores the average and current value for IoU calculation."""

    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0] * self.class_num, dtype='float64')
        self.avg = np.asarray([0] * self.class_num, dtype='float64')
        self.sum = np.asarray([0] * self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = IoUAverage.get_ious(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)

    @staticmethod
    def get_ious(logits, targets, smooth=1e-6):
        ious = []
        logits = logits.double()
        targets = targets.double()

        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union_logits = torch.sum(logits[:, class_index, :, :, :])
            union_targets = torch.sum(targets[:, class_index, :, :, :])
            union = union_logits + union_targets - inter

            iou = (inter + smooth) / (union + smooth)  # 添加平滑因子以避免除以零
            ious.append(iou.item())

        return np.asarray(ious)
