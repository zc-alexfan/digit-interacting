# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/facebookresearch/InterHand2.6M; We have made significant modification to the original code in developing DIGIT.
"""

import torch
import torch.nn as nn
import src.model.loss as myloss
import elytra.torch_utils as torch_utils
import elytra.hm_utils as hm_utils
from src.nets.unet import MiniUNet
import torch.nn.functional as F
from src.nets.layer import make_linear_layers
import src.model.segm_net as sn
import src.nets.hrnet as hrnet
from src.utils.preprocessing import load_skeleton
import os.path as op
import numpy as np
from src.utils.transforms import convert_2p5_3d_batch


class DIGIT(nn.Module):
    def __init__(self, beta, args):
        super().__init__()
        self.args = args
        self.feat_size = 64
        self.backbone = hrnet.get_pose_net()
        in_channel = 32 + 512
        num_joints = 42
        self.conv_hm = nn.Conv2d(in_channel, num_joints, 1)
        self.conv_depth = nn.Conv2d(in_channel, num_joints, 1)
        self.hand_segm = sn.SegmNet()

        self.conv_segm_feat = nn.Sequential(
            nn.Conv2d(33, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.conv_head = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.merge_layer = MiniUNet(in_channel, in_channel)

        in_dim = 512
        self.root_fc = make_linear_layers(
            [in_dim, 512, args.output_root_hm_shape], relu_final=False
        )
        self.hand_fc = make_linear_layers([in_dim, 512, 2], relu_final=False)

        self.beta = beta

    def defrost_all(self):
        torch_utils.toggle_parameters(self, True)
        self.frozen_backbone = False
        print(">>> Defrost ALL")

    def forward(self, img, segm_target_256, segm_valid):
        batch_size, _, im_dim, _ = img.shape
        img = self.backbone(img)
        segm_dict = self.hand_segm(img, segm_target_256, segm_valid)
        segm_logits = segm_dict["segm_logits"]
        segm_feat = self.conv_segm_feat(segm_logits)

        img = torch.cat((img, segm_feat), dim=1)
        img = self.merge_layer(img)

        hm_2d = self.conv_hm(img)
        hm_depth = self.conv_depth(img)
        head_feat = self.conv_head(img)

        num_joints = hm_2d.shape[1]

        hm_norm = hm_2d.view(batch_size * num_joints, self.feat_size * self.feat_size)
        hm_norm = F.softmax(self.beta * hm_norm, dim=1)
        hm_norm = hm_norm.view(batch_size, num_joints, self.feat_size, self.feat_size)
        hm_depth_agg = hm_norm * hm_depth
        zval = hm_depth_agg.view(batch_size, num_joints, -1).sum(dim=2)

        # reform head feat
        batch_size, dim = head_feat.shape[:2]
        head_feat = head_feat.view(batch_size, dim, -1)
        head_feat = head_feat.mean(dim=2)

        # root heatmap 1d
        root_heatmap1d = self.root_fc(head_feat)
        root_depth = self.soft_argmax_1d(root_heatmap1d).view(-1, 1)

        # hand type
        hand_type = self.hand_fc(head_feat)
        hand_type_sigmoid = torch.sigmoid(hand_type)
        out_dict = {}
        out_dict["hm_2d"] = hm_2d
        out_dict["hm_norm"] = hm_norm
        out_dict["hm_depth"] = hm_depth
        out_dict["hm_depth_agg"] = hm_depth_agg
        out_dict["zval"] = zval
        out_dict["rel_root_depth_out"] = root_depth
        out_dict["hand_type"] = hand_type
        out_dict["hand_type_sigmoid"] = hand_type_sigmoid
        out_dict["root_heatmap1d"] = root_heatmap1d
        out_dict["segm_dict"] = segm_dict
        return out_dict

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 1)

        mask = torch.cuda.comm.broadcast(
            torch.arange(self.args.output_root_hm_shape).type(torch.cuda.FloatTensor),
            devices=[heatmap1d.device.index],
        )[0]

        accu = heatmap1d * mask
        coord = accu.sum(dim=1)
        return coord


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        # modules
        self.pose_reg = DIGIT(1.0, args)

        # loss functions
        self.joint_heatmap_loss = myloss.JointHeatmapLoss()
        self.rel_root_depth_loss = myloss.RelRootDepthLoss()
        self.hand_type_loss = myloss.HandTypeLoss()
        self.joint_l1loss = myloss.JointL1Loss()
        self.segm_loss = myloss.SegmLoss()

        self.skeleton = load_skeleton(
            op.join(self.args.anno_path, "skeleton.txt"), 21 * 2
        )
        left_bones = [
            [joint_idx, sk["parent_id"]]
            for joint_idx, sk in enumerate(self.skeleton)
            if "l_" in sk["name"] and sk["parent_id"] != -1
        ]
        right_bones = [
            [joint_idx, sk["parent_id"]]
            for joint_idx, sk in enumerate(self.skeleton)
            if "r_" in sk["name"] and sk["parent_id"] != -1
        ]
        self.left_bones = np.array(left_bones)
        self.right_bones = np.array(right_bones)

    def load_pretrained(self, path):
        if path == "":
            return
        sd = torch.load(path)
        sd = {k.replace("model.", ""): v for k, v in sd.items()}
        self.load_state_dict(sd)
        print("Loaded pretrained weights: {}".format(path))

    def convert_2p5_3d(
        self,
        joint_xyz,
        inv_trans,
        abs_depth_left,
        abs_depth_right,
        focal,
        princpt,
        flipped,
        hw_sizes,
    ):

        joint_num = 21
        input_img_shape = self.args.input_img_shape
        output_hm_shape = self.args.output_hm_shape
        bbox_3d_size = self.args.bbox_3d_size

        pts3d_cam = convert_2p5_3d_batch(
            joint_num,
            joint_xyz,
            flipped,
            inv_trans,
            abs_depth_left,
            abs_depth_right,
            focal,
            princpt,
            input_img_shape,
            output_hm_shape,
            bbox_3d_size,
        )

        return pts3d_cam

    def forward(self, inputs, targets, meta_info, mode):
        loss_dict = {}

        # unpacking
        input_img = inputs["img"]
        target_joint_coord = targets["joint_coord"]
        target_rel_root_depth = targets["rel_root_depth"]
        target_hand_type = targets["hand_type"]
        joint_valid = meta_info["joint_valid"]
        root_valid = meta_info["root_valid"]
        hand_type_valid = meta_info["hand_type_valid"]
        inv_trans = meta_info["inv_trans"]
        segm_target_256 = targets["segm_256"]
        segm_valid = targets["segm_valid"]

        pose_dict = self.pose_reg(input_img, segm_target_256, segm_valid)
        hm_2d = pose_dict["hm_2d"]
        hm_norm = pose_dict["hm_norm"]
        zval = pose_dict["zval"]
        rel_root_depth_out = pose_dict["rel_root_depth_out"]
        hand_type = pose_dict["hand_type"]
        hand_type_sigmoid = pose_dict["hand_type_sigmoid"]
        segm_dict = pose_dict["segm_dict"]
        segm_dict["segm_256"] = segm_target_256
        segm_dict["segm_valid"] = segm_valid

        joint_xy = hm_utils.hm2xy(hm_2d, self.args.output_hm_shape, self.args.beta)

        # 2p5 to 3d
        joint_xyz = torch.cat((joint_xy, zval[:, :, None]), dim=2)
        focal = meta_info["focal"]
        princpt = meta_info["princpt"]
        abs_depth_left = meta_info["abs_depth_left"]
        abs_depth_right = meta_info["abs_depth_right"]
        hw_sizes = meta_info["hw_size"]
        flipped = meta_info["flipped"]

        kp_cam = self.convert_2p5_3d(
            joint_xyz,
            inv_trans,
            abs_depth_left,
            abs_depth_right,
            focal,
            princpt,
            flipped,
            hw_sizes,
        )

        gt_cam = self.convert_2p5_3d(
            target_joint_coord,
            inv_trans,
            abs_depth_left,
            abs_depth_right,
            focal,
            princpt,
            flipped,
            hw_sizes,
        )

        joint_mask = joint_valid
        root_mask = root_valid
        hand_type_mask = hand_type_valid

        # Prior losses
        left_bone_vector = (
            joint_xyz[:, self.left_bones[:, 0]] - joint_xyz[:, self.left_bones[:, 1]]
        )
        right_bone_vector = (
            joint_xyz[:, self.right_bones[:, 0]] - joint_xyz[:, self.right_bones[:, 1]]
        )
        gt_left_bone_vector = (
            target_joint_coord[:, self.left_bones[:, 0]]
            - target_joint_coord[:, self.left_bones[:, 1]]
        )
        gt_right_bone_vector = (
            target_joint_coord[:, self.right_bones[:, 0]]
            - target_joint_coord[:, self.right_bones[:, 1]]
        )

        left_bone_valid = (
            joint_valid[:, self.left_bones[:, 0]]
            * joint_valid[:, self.left_bones[:, 1]]
        )
        right_bone_valid = (
            joint_valid[:, self.right_bones[:, 0]]
            * joint_valid[:, self.right_bones[:, 1]]
        )

        left_err = (left_bone_vector - gt_left_bone_vector) ** 2
        right_err = (right_bone_vector - gt_right_bone_vector) ** 2

        left_err = (left_err * left_bone_valid[:, :, None]).mean()
        right_err = (right_err * right_bone_valid[:, :, None]).mean()

        # FS losses
        loss_dict["loss_rel_root_depth"] = self.rel_root_depth_loss(
            rel_root_depth_out, target_rel_root_depth, root_mask
        )
        loss_dict["loss_hand_type"] = self.hand_type_loss(
            hand_type, target_hand_type, hand_type_mask
        )
        loss_dict["loss_joint"] = self.joint_l1loss(
            joint_xyz, target_joint_coord, joint_mask
        ).mean()
        loss_dict["loss_segm"] = self.segm_loss(segm_dict) * 10.0
        loss_dict["loss_bone"] = 0.1 * (left_err + right_err)

        if mode == "train":
            return loss_dict

        if mode == "vis":
            vis_dict = {}
            segm_l_mask_128 = (
                F.interpolate(
                    segm_dict["segm_mask"].float()[:, None, :, :], 256, mode="nearest"
                )
                .long()
                .squeeze()
            )
            segm_r_mask_128 = segm_l_mask_128.clone()
            vis_dict["segm_l_mask"] = torch_utils.tensor2np(segm_l_mask_128)
            vis_dict["segm_r_mask"] = torch_utils.tensor2np(segm_r_mask_128)
            vis_dict["input_img"] = torch_utils.tensor2np(input_img)
            vis_dict["segm_target_256"] = torch_utils.tensor2np(
                F.interpolate(segm_target_256.float(), 256, mode="nearest").long()
            )
            vis_dict["gt_joint_2p5"] = torch_utils.tensor2np(target_joint_coord)
            vis_dict["joint_valid"] = torch_utils.tensor2np(joint_valid)
            vis_dict["im_path"] = meta_info["im_path"]
            vis_dict["pred_joint_2p5"] = torch_utils.tensor2np(joint_xyz)
            vis_dict["hm_2d"] = torch_utils.tensor2np(hm_2d)
            vis_dict["segm_valid"] = torch_utils.tensor2np(segm_valid)

            vis_dict["pred_cam"] = torch_utils.tensor2np(kp_cam)
            vis_dict["gt_cam"] = torch_utils.tensor2np(gt_cam)
            vis_dict["hm_norm"] = torch_utils.tensor2np(hm_norm)
            return vis_dict

        loss_dict = {k: loss_dict[k].mean() for k in loss_dict}
        loss_dict["total_loss"] = sum(loss_dict[k] for k in loss_dict)

        # test
        out_dict = {
            "joint_coord": joint_xyz,
            "rel_root_depth": rel_root_depth_out,
            "hand_type": hand_type_sigmoid,
            "inv_trans": inv_trans,
            "idx": meta_info["idx"],
        }
        return out_dict, loss_dict


def get_model_pl(args):
    model = Model(args)
    return model
