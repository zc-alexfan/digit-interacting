import torch
import torch.nn as nn
import src.nets.layer as layer


class SegmHead(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, class_dim):
        super().__init__()

        # upsample features
        self.upsampler = layer.UpSampler(in_dim, hidden_dim1, hidden_dim2)

        segm_net = layer.DoubleConv(hidden_dim2, class_dim)
        segm_net.double_conv = segm_net.double_conv[:4]
        self.segm_net = segm_net

    def forward(self, img_feat):
        # feature up sample to 256
        hr_img_feat = self.upsampler(img_feat)
        segm_logits = self.segm_net(hr_img_feat)
        return {'segm_logits': segm_logits}


class SegmNet(nn.Module):
    def __init__(self):
        super(SegmNet, self).__init__()
        self.segm_head = SegmHead(32, 128, 64, 33)

    def map2labels(self, segm_hand):
        with torch.no_grad():
            segm_hand = segm_hand.permute(0, 2, 3, 1)
            _, pred_segm_hand = segm_hand.max(dim=3)
            return pred_segm_hand

    def forward(self, img_feat, segm_target_256, segm_valid):
        segm_dict = self.segm_head(img_feat)
        segm_logits = segm_dict['segm_logits']

        segm_mask = self.map2labels(segm_logits)

        segm_dict['segm_mask'] = segm_mask
        segm_dict['segm_logits'] = segm_logits
        return segm_dict


