import torch
import torch.nn as nn
from torch.nn import functional as F


class JointHeatmapLoss(nn.Module):
    def __ini__(self):
        super(JointHeatmapLoss, self).__init__()

    def forward(self, joint_out, joint_gt, joint_valid):
        loss = (joint_out - joint_gt)**2 * joint_valid[:,:,None,None,None]
        return loss


class JointL1Loss(nn.Module):
    def __init__(self):
        super(JointL1Loss, self).__init__()
        self.criterion = nn.L1Loss(reduction='none')

    def forward(self, joint_out, joint_gt, joint_valid):
        loss = self.criterion(joint_out, joint_gt) * joint_valid[:, :, None]
        return loss


class HandTypeLoss(nn.Module):
    def __init__(self):
        super(HandTypeLoss, self).__init__()

    def forward(self, hand_type_out, hand_type_gt, hand_type_valid):
        loss = F.binary_cross_entropy_with_logits(hand_type_out, hand_type_gt, reduction='none')
        loss = loss.mean(1)
        loss = loss * hand_type_valid

        return loss

class RelRootDepthLoss(nn.Module):
    def __init__(self):
        super(RelRootDepthLoss, self).__init__()

    def forward(self, root_depth_out, root_depth_gt, root_valid):
        loss = torch.abs(root_depth_out - root_depth_gt) * root_valid
        return loss


class SegmLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, data_dict):
        segm_256 = data_dict['segm_256']
        segm_valid = data_dict['segm_valid']
        segm_target = segm_256[:, 0]

        segm_logits = data_dict['segm_logits']

        # segm loss
        dist_hand = self.loss(segm_logits, segm_target)*segm_valid[:, None, None]
        total_loss = dist_hand.mean().view(-1)
        return total_loss
