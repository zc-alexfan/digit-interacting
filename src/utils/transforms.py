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
from elytra.tf_utils import pixel2cam_batch


def pixel2cam_torch(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = torch.stack((x, y, z), dim=1)
    return cam_coord


def trans_point2d_torch(joints_xy, inv_trans):
    num_pts = joints_xy.shape[0]
    joints_xy_homo = torch.ones((num_pts, 3))
    joints_xy_homo[:, :2] = joints_xy
    joints_xy_inv = torch.mm(joints_xy_homo, inv_trans.transpose(1, 0))
    return joints_xy_inv

"""
Convert 2.5D Joints to 3D Joints
Undo the data augmentation on xy
Add the absolute depth for each hand
The final 3D joints are in the camera space.
"""
def convert_2p5_3d_torch(
    joint_num, kpt_2p5, inv_trans, abs_depth_left, abs_depth_right,
    focal, princpt, input_img_shape, output_hm_shape, bbox_3d_size):
    assert isinstance(kpt_2p5, torch.Tensor)
    assert kpt_2p5.shape == (joint_num*2, 3)

    joint_type = {
        'right': torch.arange(0, joint_num),
        'left': torch.arange(joint_num, joint_num*2)}
    kpt_2p5[:, 0] = kpt_2p5[:, 0]/output_hm_shape[2]*input_img_shape[1]
    kpt_2p5[:, 1] = kpt_2p5[:, 1]/output_hm_shape[1]*input_img_shape[0]
    kpt_2p5[:, 2] = (kpt_2p5[:, 2]/output_hm_shape[0] * 2 - 1) * (bbox_3d_size/2)

    kpt_2p5[:, :2] = trans_point2d_torch(kpt_2p5[:, :2], inv_trans)

    # add root joint depth
    kpt_2p5[joint_type['right'],2] += abs_depth_right
    kpt_2p5[joint_type['left'],2] += abs_depth_left 

    # back project to camera coordinate system
    joint_cam = pixel2cam_torch(kpt_2p5, focal, princpt)
    return joint_cam


def trans_point2d_batch(joints_xy, inv_trans):
    """
    Perform affine transformation to 2d points.
    The affine transformation is encoded in inv_trans
    Inputs:
        joints_xy: (batch, points, 2)
        inv_trans: (batch, 2, 3)
    Outputs:
        joints_xy: (batch, points, 2)
    """
    batch_size = joints_xy.shape[0]
    num_pts = joints_xy.shape[1]
    assert joints_xy.shape[0] == batch_size
    assert joints_xy.shape[2] == 2
    assert inv_trans.shape == (batch_size, 2, 3)
    dev = joints_xy.device
    joints_xy_homo = torch.ones((batch_size, num_pts, 3)).to(dev)
    joints_xy_homo[:, :, :2] = joints_xy
    joints_xy_inv = torch.bmm(joints_xy_homo, inv_trans.permute(0, 2, 1))
    return joints_xy_inv



def unflip_entries(mypts3d_cam, flipped):
    pts3d_cam = mypts3d_cam.clone()
    assert isinstance(pts3d_cam, torch.Tensor)
    assert len(pts3d_cam.shape) >= 2
    assert isinstance(flipped, torch.Tensor)
    batch_size, joint_num = pts3d_cam.shape[:2]
    joint_num = int(joint_num/2)
    assert flipped.shape[0] == batch_size

    dev = pts3d_cam.device
    joint_type = {
        'right': torch.arange(0, joint_num).to(dev),
        'left': torch.arange(joint_num, joint_num*2).to(dev)}
    flipped_idx = torch.nonzero(flipped, as_tuple=True)[0]
    tmp = pts3d_cam[flipped_idx]
    tmp[:, joint_type['right']], tmp[:, joint_type['left']] = \
        tmp[:, joint_type['left']].clone(), tmp[:, joint_type['right']].clone()
    pts3d_cam[flipped_idx] = tmp
    return pts3d_cam


def convert_2p5_3d_batch(
        joint_num, mykpt_2p5, flipped,
        inv_trans, myabs_depth_left, myabs_depth_right,
        focal, princpt, input_img_shape, output_hm_shape, bbox_3d_size):
    """
    Note: this version does not do unflipping
    """
    kpt_2p5 = mykpt_2p5.clone()
    abs_depth_left = myabs_depth_left.clone()
    abs_depth_right = myabs_depth_right.clone()

    assert isinstance(kpt_2p5, torch.Tensor)
    batch_size = kpt_2p5.shape[0]
    assert kpt_2p5.shape[1:] == (joint_num*2, 3)
    assert inv_trans.shape == (batch_size, 2, 3)
    assert abs_depth_left.shape[0] == batch_size
    assert abs_depth_left.shape == abs_depth_right.shape
    assert focal.shape == (batch_size, 2)
    assert princpt.shape == (batch_size, 2)
    dev = kpt_2p5.device
    joint_type = {
        'right': torch.arange(0, joint_num).to(dev),
        'left': torch.arange(joint_num, joint_num*2).to(dev)}

    # map from hm scale to image scale
    kpt_2p5[:, :, 0] = kpt_2p5[:, :, 0]/output_hm_shape[2]*input_img_shape[1]
    kpt_2p5[:, :, 1] = kpt_2p5[:, :, 1]/output_hm_shape[1]*input_img_shape[0]
    kpt_2p5[:, :, 2] = (kpt_2p5[:, :, 2]/output_hm_shape[0] * 2 - 1) * (bbox_3d_size/2)

    # undo affine transform for xy of keypoints in image scale
    # after this, the keypoints are in the non-"cropped, scale, rot" original space
    # a.k.a the original image space
    kpt_2p5[:, :, :2] = trans_point2d_batch(kpt_2p5[:, :, :2], inv_trans)

    # identify flipped poses
    flipped_idx = torch.nonzero(flipped, as_tuple=True)[0]

    # flip poses use the opposite depths
    tmp = abs_depth_right[flipped_idx]
    abs_depth_right[flipped_idx] = abs_depth_left[flipped_idx]
    abs_depth_left[flipped_idx] = tmp

    # add depth
    kpt_2p5[:, joint_type['right'], 2] += abs_depth_right[:, None]
    kpt_2p5[:, joint_type['left'], 2] += abs_depth_left[:, None]

    # normal backprojection that does not rely on affine transformation
    joint_cam = pixel2cam_batch(kpt_2p5, focal, princpt)
    return joint_cam
