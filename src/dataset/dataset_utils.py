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

import numpy as np
import torch
import os.path as osp
from src.utils.preprocessing import process_bbox
from elytra.tf_utils_np import world2cam, cam2pixel


def downsample(raw_data, split):
    if 'small' not in split and 'mini' not in split:
        return raw_data
    import random
    random.seed(1)
    assert random.randint(0, 100) == 17, \
        "Same seed but different results; Subsampling might be different."
    all_keys = list(raw_data.keys())
    if split == 'minival' or split == 'minitest':
        num_samples = 1000
    elif split == 'smallval' or split == 'smalltest':
        num_samples = int(0.25*len(all_keys))
    elif split == 'minitrain':
        num_samples = 1000
    elif split == 'smalltrain':
        num_samples = int(0.1*len(all_keys))
    elif split == 'tinytrain':
        num_samples = int(0.05*len(all_keys))
    else:
        assert False, "Unknown split {}".format(split)
    curr_keys = random.sample(all_keys, num_samples)

    new_anns = {}
    for key in curr_keys:
        new_anns[key] = raw_data[key]
    return new_anns


def process_anno(
        img, cameras, bbox_rootnet, abs_depth_rootnet, joints, trans_test,
        input_img_shape, joint_num, joint_type, root_joint_idx, img_path, mode):
    ann = img['anno']

    capture_id = img['capture']
    seq_name = img['seq_name']
    cam = img['camera']
    frame_idx = img['frame_idx']
    img_path = osp.join(img_path, mode, img['file_name'])

    # extrinsics
    campos = np.array(
        cameras[str(capture_id)]['campos'][str(cam)],
        dtype=np.float32)
    camrot = np.array(
        cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)

    # instrinsics
    focal = np.array(
        cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32)
    princpt = np.array(
        cameras[str(capture_id)]['princpt'][str(cam)],
        dtype=np.float32)

    # hand joints in world coord
    joint_world = np.array(
        joints[str(capture_id)][str(frame_idx)]['world_coord'],
        dtype=np.float32)

    # hand joints in cam coord
    joint_cam = world2cam(
        joint_world.transpose(1, 0), camrot, campos.reshape(3, 1)
        ).transpose(1, 0)

    # project joints in cam coord to pixel space (3d->2d)
    joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

    joint_valid = np.array(
        ann['joint_valid'], dtype=np.float32
        ).reshape(joint_num*2)

    # if root is not valid -> root-relative 3D pose is also not valid.
    # Therefore, mark all joints as invalid
    joint_valid[joint_type['right']] *= joint_valid[root_joint_idx['right']]
    joint_valid[joint_type['left']] *= joint_valid[root_joint_idx['left']]

    hand_type = ann['hand_type']
    hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)

    # the absolute root depth of the left/right hands
    # are either predicted by root net or we use the gt.
    if (mode == 'val' or mode == 'test')\
            and trans_test == 'rootnet':
        bbox = bbox_rootnet
        abs_depth = abs_depth_rootnet
    else:
        img_width, img_height = img['width'], img['height']
        # 2d bbox, not 3d
        bbox = np.array(ann['bbox'], dtype=np.float32)  # x,y,w,h
        bbox = process_bbox(bbox, (img_height, img_width), input_img_shape)
        abs_depth = {
            'right': joint_cam[root_joint_idx['right'], 2],
            'left': joint_cam[root_joint_idx['left'], 2]}

    cam_param = {'focal': focal, 'princpt': princpt}
    joint = {
        'cam_coord': joint_cam, 'img_coord': joint_img,
        'valid': joint_valid}
    data = {
        'img_path': img_path, 'seq_name': seq_name,
        'cam_param': cam_param, 'bbox': bbox, 'joint': joint,
        'hand_type': hand_type, 'hand_type_valid': hand_type_valid,
        'abs_depth': abs_depth, 'file_name': img['file_name'],
        'capture': capture_id, 'cam': cam, 'frame': frame_idx}
    return data


def swap_lr_labels_segm_target_channels(segm_target):
    """
    Flip left and right label (not the width) of a single segmentation image.
    """
    assert isinstance(segm_target, torch.Tensor)
    assert len(segm_target.shape) == 3

    assert segm_target.min() >= 0
    assert segm_target.max() <= 32
    img_segm = segm_target.clone()
    right_idx = ((1 <= img_segm)*(img_segm <= 16)).nonzero(as_tuple=True)
    left_idx = ((17 <= img_segm)*(img_segm <= 32)).nonzero(as_tuple=True)
    img_segm[right_idx[0], right_idx[1], right_idx[2]] += 16
    img_segm[left_idx[0], left_idx[1], left_idx[2]] -= 16
    img_segm_swapped = img_segm.clone()
    img_segm_swapped[1], img_segm_swapped[2] = img_segm_swapped[2].clone(), img_segm_swapped[1].clone()
    return img_segm_swapped
