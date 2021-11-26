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

import copy
import numpy as np
from elytra.tf_utils_np import cam2pixel, pixel2cam
from src.utils.preprocessing import trans_point2d


def skeleton_to_bone_dict(skeleton):
    left_bones = np.array(
        [(jidx, skel['parent_id']) for jidx, skel in enumerate(skeleton)
            if 'l_' in skel['name'] and skel['parent_id'] != -1]
    )

    right_bones = np.array(
        [(jidx, skel['parent_id']) for jidx, skel in enumerate(skeleton)
            if 'r_' in skel['name'] and skel['parent_id'] != -1]
    )

    left_bone_names = []
    for from_jidx, to_jidx in left_bones.tolist():
        left_bone_names.append((skeleton[from_jidx]['name'],
                                skeleton[to_jidx]['name']))

    right_bone_names = []
    for from_jidx, to_jidx in right_bones.tolist():
        right_bone_names.append((skeleton[from_jidx]['name'],
                                 skeleton[to_jidx]['name']))

    bone_dict = {}
    bone_dict['left_bone_idx'] = left_bones
    bone_dict['right_bone_idx'] = right_bones
    bone_dict['left_bone_names'] = left_bone_names
    bone_dict['right_bone_names'] = right_bone_names
    return bone_dict


def undo_transform_2p5(
        preds_2p5, curr_inv,
        hm_shape, img_shape, bbox_size, joint_num):
    pred_2p5_img = preds_2p5.copy()
    pred_2p5_img[:, 0] = pred_2p5_img[:, 0]/hm_shape[2]*img_shape[1]
    pred_2p5_img[:, 1] = pred_2p5_img[:, 1]/hm_shape[1]*img_shape[0]
    for j in range(joint_num*2):
        pred_2p5_img[j, :2] = trans_point2d(pred_2p5_img[j, :2], curr_inv)
    pred_2p5_img[:, 2] = (pred_2p5_img[:, 2]/hm_shape[0] * 2 - 1) * (bbox_size/2)
    return pred_2p5_img


def convert_2p5_3d_np(
        pred_2p5_img, pred_rel_depth,
        joint_valid, gt_hand_type,
        left_idx, right_idx,
        root_idx_left, root_idx_right,
        abs_depth_left, abs_depth_right, focal, princpt,
        root_hm_shape, bbox_root_size):

    pred_3d = pred_2p5_img.copy()
    if gt_hand_type == 'interacting' and \
            joint_valid[root_idx_left] and joint_valid[root_idx_right]:
        pred_rel_root_depth = (pred_rel_depth/root_hm_shape * 2 - 1) * (
            bbox_root_size/2)
        depth_left = abs_depth_right + pred_rel_root_depth
        depth_right = abs_depth_right
    else:
        depth_left = abs_depth_left
        depth_right = abs_depth_right

    pred_3d[left_idx, 2] += depth_left
    pred_3d[right_idx, 2] += depth_right
    pred_3d_cam = pixel2cam(pred_3d, focal, princpt)
    return pred_3d_cam


def update_bone_metrics(
        pred_cam, bone_metric_list, left_bone_idx, right_bone_idx):
    left_bones = pred_cam[left_bone_idx[:, 0]] - pred_cam[left_bone_idx[:, 1]]
    right_bones = pred_cam[right_bone_idx[:, 0]] - pred_cam[right_bone_idx[:, 1]]
    left_bone_len = np.sqrt((left_bones**2).sum(axis=1))
    right_bone_len = np.sqrt((right_bones**2).sum(axis=1))

    per_bone_len_diff = np.abs(left_bone_len - right_bone_len)
    bone_errors = per_bone_len_diff.tolist()

    for bone_err, bone_container in zip(bone_errors, bone_metric_list):
        bone_container.append(bone_err)
    return bone_metric_list


def update_mrrpe(
        pred_2p5_img, gt_joint, mrrpe, abs_depth_right, pred_rel_depth,
        gt_hand_type, joint_valid, focal, princpt,
        root_idx_left, root_idx_right, bbox_root_size, root_hm_shape):

    if gt_hand_type == 'interacting' and \
            joint_valid[root_idx_left] and joint_valid[root_idx_right]:
        pred_rel_root_depth = (pred_rel_depth/root_hm_shape * 2 - 1) * (
                bbox_root_size/2)

        pred_left_root_img = pred_2p5_img[root_idx_left].copy()
        pred_left_root_img[2] += abs_depth_right + pred_rel_root_depth
        pred_left_root_cam = pixel2cam(
                pred_left_root_img[None, :], focal, princpt)[0]

        pred_right_root_img = pred_2p5_img[root_idx_right].copy()
        pred_right_root_img[2] += abs_depth_right
        pred_right_root_cam = pixel2cam(
                pred_right_root_img[None, :], focal, princpt)[0]

        pred_rel_root = pred_left_root_cam - pred_right_root_cam
        gt_rel_root = gt_joint[root_idx_left] - gt_joint[root_idx_right]
        mrrpe.append(float(np.sqrt(np.sum((pred_rel_root - gt_rel_root)**2))))
    return mrrpe


def subtract_root(
        joint_coord, root_idx_left, root_idx_right,
        joint_idx_left, joint_idx_right):
    joint_coord[joint_idx_left] -= joint_coord[root_idx_left, None, :]
    joint_coord[joint_idx_right] -= joint_coord[root_idx_right, None, :]
    return joint_coord


def update_hand_cls_counts(
        hand_type_valid, preds_hand_type, gt_hand_type,
        acc_hand_cls, hand_cls_cnt):
    # handedness accuray
    if hand_type_valid:
        if gt_hand_type == 'right' and \
                preds_hand_type[0] > 0.5 and preds_hand_type[1] < 0.5:
            acc_hand_cls += 1
        elif gt_hand_type == 'left' and \
                preds_hand_type[0] < 0.5 and preds_hand_type[1] > 0.5:
            acc_hand_cls += 1
        elif gt_hand_type == 'interacting' and \
                preds_hand_type[0] > 0.5 and preds_hand_type[1] > 0.5:
            acc_hand_cls += 1
        hand_cls_cnt += 1
    return acc_hand_cls, hand_cls_cnt


def update_mpjpe(
        pred_2p5_img, gt_joint, joint_valid, gt_hand_type,
        mpjpe_sh, mpjpe_ih, mpjpe_all,
        joint_idx_left, joint_idx_right, abs_depth_left, abs_depth_right,
        focal, princpt, root_idx_left, root_idx_right, joint_num):

    # add root joint depth
    pred_2p5_img[joint_idx_right, 2] += abs_depth_right
    pred_2p5_img[joint_idx_left, 2] += abs_depth_left

    # back project to camera coordinate system
    pred_joint_cam = pixel2cam(pred_2p5_img, focal, princpt)

    # root joint alignment
    pred_joint_cam = subtract_root(
            pred_joint_cam,
            root_idx_left, root_idx_right,
            joint_idx_left, joint_idx_right)
    gt_joint = subtract_root(
            gt_joint,
            root_idx_left, root_idx_right,
            joint_idx_left, joint_idx_right)

    # mpjpe
    for j in range(joint_num*2):
        if joint_valid[j]:
            mpjpe_val = np.sqrt(np.sum((pred_joint_cam[j] - gt_joint[j])**2))
            if gt_hand_type == 'right' or gt_hand_type == 'left':
                mpjpe_sh[j].append(mpjpe_val)
            else:
                mpjpe_ih[j].append(mpjpe_val)
            mpjpe_all[j].append(mpjpe_val)
        else:
            mpjpe_all[j].append(None)
    return mpjpe_sh, mpjpe_ih, mpjpe_all


def compute_metric_dict(
        acc_hand_cls, hand_cls_cnt, mrrpe,
        joint_num, mpjpe_sh, mpjpe_ih,
        per_bone_diff_list, bone_dict,
        skeleton, verbose):
    metric_dict = {}
    if hand_cls_cnt > 0:
        metric_dict['handness_acc'] = acc_hand_cls/hand_cls_cnt
        if verbose:
            print('Handedness accuracy: ' + str(metric_dict['handness_acc']))
    if len(mrrpe) > 0:
        metric_dict['mrrpe'] = sum(mrrpe)/len(mrrpe)
        if verbose:
            print('MRRPE: ' + str(metric_dict['mrrpe']))

    tot_err = []
    eval_summary = 'MPJPE for each joint: \n'
    for j in range(joint_num*2):
        tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j]))))
        joint_name = skeleton[j]['name']
        eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
        tot_err.append(tot_err_j)
        metric_dict[joint_name + '_all'] = tot_err_j
    metric_dict['mpjpe_all'] = np.mean(tot_err)

    if verbose:
        print(eval_summary)
        print('MPJPE for all hand sequences: %.2f' % (
            metric_dict['mpjpe_all']))
        print()

    eval_summary = 'MPJPE for each joint: \n'
    for j in range(joint_num*2):
        mpjpe_sh[j] = np.mean(np.stack(mpjpe_sh[j]))
        joint_name = skeleton[j]['name']
        eval_summary += (joint_name + ': %.2f, ' % mpjpe_sh[j])
        metric_dict[joint_name + '_sh'] = mpjpe_sh[j]

    metric_dict['mpjpe_sh'] = np.mean(mpjpe_sh)
    if verbose:
        print(eval_summary)
        print('MPJPE for single hand sequences: %.2f' % (
            metric_dict['mpjpe_sh']))
        print()

    eval_summary = 'MPJPE for each joint: \n'
    for j in range(joint_num*2):
        mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
        joint_name = skeleton[j]['name']
        eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih[j])
        metric_dict[joint_name + '_ih'] = mpjpe_ih[j]
    metric_dict['mpjpe_ih'] = np.mean(mpjpe_ih)
    if verbose:
        print(eval_summary)
        print('MPJPE for interacting hand sequences: %.2f' % (
            metric_dict['mpjpe_ih']))
    return metric_dict



def evaluate(
        cfg, preds, datalist, joint_num,
        root_joint_idx, joint_type, skeleton, verbose=True):
    if verbose:
        print('\nEvaluation start...')
    gts = copy.deepcopy(datalist)
    preds_2p5 = preds['joint_coord']
    preds_rel_root_depth = preds['rel_root_depth']
    preds_hand_type = preds['hand_type']
    inv_trans = preds['inv_trans']
    gt_idx = preds['idx'].tolist()
    joint_idx_left = joint_type['left']
    joint_idx_right = joint_type['right']
    root_idx_left = root_joint_idx['left']
    root_idx_right = root_joint_idx['right']

    hm_shape = cfg.output_hm_shape
    img_shape = cfg.input_img_shape
    root_hm_shape = cfg.output_root_hm_shape
    bbox_root_size = cfg.bbox_3d_size_root
    bbox_size = cfg.bbox_3d_size

    sample_num = len(preds_2p5)

    mpjpe_sh = [[] for _ in range(joint_num*2)]
    mpjpe_ih = [[] for _ in range(joint_num*2)]
    mpjpe_all = [[] for _ in range(joint_num*2)]

    mrrpe = []
    acc_hand_cls = 0
    hand_cls_cnt = 0

    out_pred_joint_cam = []
    out_gt_joint_cam = []
    out_pred_joint_2d = []
    out_gt_joint_2d = []
    out_meta = []
    im_path_list = []
    frame_type_list = []

    bone_dict = skeleton_to_bone_dict(skeleton)
    left_bone_idx = bone_dict['left_bone_idx']
    right_bone_idx = bone_dict['right_bone_idx']
    per_bone_diff_list = [[] for _ in range(20)]

    for n in range(sample_num):
        data = gts[gt_idx[n]]
        curr_inv = inv_trans[n]
        curr_pred_rel_depth = preds_rel_root_depth[n]
        curr_2p5 = preds_2p5[n]

        # unpack data
        focal = data['cam_param']['focal']
        princpt = data['cam_param']['princpt']
        img_path = data['img_path']
        gt_joint = data['joint']['cam_coord']
        joint_valid = data['joint']['valid']
        gt_hand_type = data['hand_type']
        frame_type_list.append(gt_hand_type)
        hand_type_valid = data['hand_type_valid']
        abs_depth_left = data['abs_depth']['left']
        abs_depth_right = data['abs_depth']['right']

        # pack meta dict
        meta_dict = {}
        meta_dict['img_path'] = img_path
        meta_dict['joint_valid'] = joint_valid
        out_meta.append(meta_dict)
        im_path_list.append(img_path)

        # 2d keypoints
        gt_2d_img = cam2pixel(
                gt_joint.copy(), focal, princpt)[:, :2]
        pred_2p5_img = undo_transform_2p5(
                curr_2p5, curr_inv,
                hm_shape, img_shape, bbox_size, joint_num)
        out_pred_joint_2d.append(pred_2p5_img[:, :2].copy())
        out_gt_joint_2d.append(gt_2d_img)

        pred_cam = convert_2p5_3d_np(
                pred_2p5_img, curr_pred_rel_depth,
                joint_valid, gt_hand_type,
                joint_idx_left, joint_idx_right,
                root_idx_left, root_idx_right,
                abs_depth_left, abs_depth_right,
                focal, princpt, root_hm_shape, bbox_root_size)
        out_pred_joint_cam.append(pred_cam)
        out_gt_joint_cam.append(gt_joint.copy())
        if gt_hand_type == 'interacting':
            update_bone_metrics(
                    pred_cam, per_bone_diff_list, left_bone_idx, right_bone_idx)

        # compute mrrpe
        mrrpe = update_mrrpe(
            pred_2p5_img, gt_joint, mrrpe,
            abs_depth_right, curr_pred_rel_depth,
            gt_hand_type, joint_valid, focal, princpt,
            root_idx_left, root_idx_right, bbox_root_size, root_hm_shape)

        # compute mpjpe
        mpjpe_sh, mpjpe_ih, mpjpe_all = update_mpjpe(
            pred_2p5_img, gt_joint, joint_valid,
            gt_hand_type, mpjpe_sh, mpjpe_ih, mpjpe_all,
            joint_idx_left, joint_idx_right, abs_depth_left, abs_depth_right,
            focal, princpt, root_idx_left, root_idx_right, joint_num)

        acc_hand_cls, hand_cls_cnt = update_hand_cls_counts(
            hand_type_valid, preds_hand_type[n], gt_hand_type,
                   acc_hand_cls, hand_cls_cnt)

    # end of loop
    out_pred_joint_cam = np.stack(out_pred_joint_cam)
    out_gt_joint_cam = np.stack(out_gt_joint_cam)
    out_pred_joint_2d = np.stack(out_pred_joint_2d)
    out_gt_joint_2d = np.stack(out_gt_joint_2d)

    out_dict = {
        'pred_joint_cam': out_pred_joint_cam,
        'gt_joint_cam': out_gt_joint_cam,
        'pred_joint_2d': out_pred_joint_2d,
        'gt_joint_2d': out_gt_joint_2d,
        'meta': out_meta
    }

    metric_dict = compute_metric_dict(
        acc_hand_cls, hand_cls_cnt, mrrpe,
        joint_num, mpjpe_sh, mpjpe_ih,
        per_bone_diff_list, bone_dict,
        skeleton, verbose)

    return (metric_dict, out_dict)
