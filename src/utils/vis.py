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
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import elytra.vis_utils as vis_utils
import math
import torch
upsampler = torch.nn.Upsample(
        size=(256, 256), mode='bilinear', align_corners=True)


def get_keypoint_rgb(skeleton):
    rgb_dict= {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        if joint_name.endswith('thumb_null'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith('thumb0'):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith('index_null'):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith('middle_null'):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith('ring_null'):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith('pinky_null'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)
        
    return rgb_dict

def vis_keypoints(
        img, kps, score, skeleton, filename,
        score_thr=0.4, line_width=2, circle_rad=2, vis_dir=None):
    
    rgb_dict = get_keypoint_rgb(skeleton)
    _img = img
    draw = ImageDraw.Draw(_img)
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']
        
        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))
        if i >= 21:
            color_scale = 0.5
        else:
            color_scale = 1.0
        bone_rgb = rgb_dict[parent_joint_name]
        bone_rgb = tuple([int(color*color_scale) for color in bone_rgb])


        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=bone_rgb, width=line_width)
        if score[i] > score_thr:
            draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=rgb_dict[joint_name])
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=rgb_dict[parent_joint_name])

    if vis_dir is None:
        return _img
    else:
        _img.save(osp.join(vis_dir, filename))


def plot_3d_views(
        my_joint_3d, joint_valid,
        frame_idx, skeleton, fix_limits, gap=20):
    joint_3d = my_joint_3d.copy()
    fig = plt.figure(figsize=(20, 5))
    num_views = my_joint_3d.shape[1]

    ax_vec = []
    xlim_vec = []
    ylim_vec = []
    zlim_vec = []
    for view_idx in range(num_views):
        ax = plt.subplot(
            1, num_views, view_idx+1, projection='3d')
        vis_3d_keypoints(
            joint_3d[frame_idx, view_idx],
            joint_valid[frame_idx, view_idx], skeleton, None, ax)
        ax_vec.append(ax)
        xlim_vec.append(ax.get_xlim())
        ylim_vec.append(ax.get_ylim())
        zlim_vec.append(ax.get_zlim())

    if fix_limits:
        xlim_min, xlim_max = zip(*xlim_vec)
        ylim_min, ylim_max = zip(*ylim_vec)
        zlim_min, zlim_max = zip(*zlim_vec)

        xlim = [min(xlim_min), max(xlim_max)]
        ylim = [min(ylim_min), max(ylim_max)]
        zlim = [min(zlim_min), max(zlim_max)]

        for ax in ax_vec:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

    im = vis_utils.fig2img(fig)
    plt.close()
    return im


def vis_3d_keypoints(
        kps_3d, score, skeleton, filename, ax, is_pred=False,
        score_thr=0.4, line_width=3, circle_rad=3,
        vis_dir=None, figsize=(5, 10)):

    rgb_dict = get_keypoint_rgb(skeleton)
    num_joints = len(skeleton)/2
    brightness = 0.8

    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i, 0], kps_3d[pid, 0]])
        y = np.array([kps_3d[i, 1], kps_3d[pid, 1]])
        z = np.array([kps_3d[i, 2], kps_3d[pid, 2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            line_color = np.array(rgb_dict[parent_joint_name])/255.
            # left hand darker
            if i > num_joints:
                line_color *= brightness
            ax.plot(
                x, z, -y, c=line_color,
                linewidth=line_width)
        if score[i] > score_thr:
            ax.scatter(
                kps_3d[i, 0], kps_3d[i, 2], -kps_3d[i, 1],
                c=np.array(rgb_dict[joint_name]).reshape(1, 3)/255.,
                marker='o')
        if score[pid] > score_thr and pid != -1:
            ax.scatter(
                kps_3d[pid, 0], kps_3d[pid, 2], -kps_3d[pid, 1],
                c=np.array(rgb_dict[parent_joint_name]).reshape(1, 3)/255.,
                marker='o')

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    vis_utils.axis_equal_3d(ax)


def generate_vis_images(
        im, gt_joint_2d, pred_joint_2d, gt_joint_cam, pred_joint_cam,
        joint_valid, skeleton, degrees=[0, 90]):
    im_list = []
    im_2d_gt = vis_keypoints(
        im.copy(), gt_joint_2d, joint_valid, skeleton, None,
        score_thr=0.4, line_width=2, circle_rad=2, vis_dir=None)
    im_2d_pred = vis_keypoints(
        im.copy(), pred_joint_2d, joint_valid, skeleton, None,
        score_thr=0.4, line_width=2, circle_rad=2, vis_dir=None)
    im_2d = vis_utils.stack_pil_images([im_2d_gt, im_2d_pred])
    im_list.append(im_2d)

    # creating 3d visualization
    fig = plt.figure(figsize=(4, 10))
    ax_gt = plt.subplot(3, 1, 1, projection='3d')
    vis_3d_keypoints(
        gt_joint_cam, joint_valid, skeleton, None, ax_gt,
        score_thr=0.5, line_width=3, circle_rad=3, vis_dir=None, figsize=(8, 10), is_pred=True)
    xlim = ax_gt.get_xlim()
    ylim = ax_gt.get_ylim()
    zlim = ax_gt.get_zlim()
    plt.gca().invert_xaxis()

    ax_pred = plt.subplot(3, 1, 2, projection='3d')
    vis_3d_keypoints(
        pred_joint_cam, joint_valid, skeleton, None, ax_pred,
        score_thr=0.5, line_width=3, circle_rad=3, vis_dir=None, figsize=(8, 10), is_pred=True)
    ax_pred.set_xlim(xlim)
    ax_pred.set_ylim(ylim)
    ax_pred.set_zlim(zlim)
    plt.gca().invert_xaxis()

    for degree in degrees:
        ax_gt.view_init(0, degree)
        ax_pred.view_init(0, degree)
        plt.tight_layout()
        im_list.append(vis_utils.fig2img(fig))

    ax_gt.view_init(90, 0)
    ax_pred.view_init(90, 0)
    plt.tight_layout()
    im_list.append(vis_utils.fig2img(fig))
    plt.close()
    return im_list

def numpy2pil(img_np):
    img_np = np.moveaxis(img_np, (0, 1, 2), (2, 0, 1))
    img_np -= img_np.min()
    img_np /= img_np.max()
    img_np *= 255
    img_np = img_np.astype(np.uint8)
    im = Image.fromarray(img_np)
    return im


def heatmap_to_input_size(xy, hm_size, input_size):
    xy /= float(hm_size)
    xy *= float(input_size)
    return xy


def plot_2d_views(
        my_joint_2p5, input_img, joint_valid,
        frame_idx, skeleton, hm_size, input_size):
    joint_2p5 = my_joint_2p5.copy()
    xy = joint_2p5[frame_idx]
    img_np = input_img[frame_idx]
    joint_v = joint_valid[frame_idx]
    im = numpy2pil(img_np)
    xy = heatmap_to_input_size(xy, hm_size, input_size)
    im = vis_keypoints(im, xy, joint_v, skeleton, None)
    return im


def plot_hm_img(
        hm, img, pred_xy_hm, jt_list, skeleton,
        num_cols=8, figsize=(40, 15), beta=0.5):
    hm = torch.FloatTensor(hm)
    img = torch.FloatTensor(img)
    fig = plt.figure(figsize=figsize)
    num_plots = len(jt_list)
    num_rows = math.ceil(1.0*num_plots/num_cols)
    for ax_idx, jt_idx in enumerate(jt_list):
        xy = pred_xy_hm[jt_idx]
        ax = plt.subplot(num_rows, num_cols, ax_idx+1)
        curr_hm = hm[jt_idx][None, None, :, :].clone()
        curr_img = img.clone().permute(1, 2, 0)
        hm_256 = upsampler(curr_hm).squeeze()
        hm_256 = hm_256.clamp(0)
        hm_256 /= hm_256.max()
        curr_img[:, :, 1] = (1.0 - beta)*curr_img[:, :, 1] + beta*hm_256*256
        curr_img = curr_img.clamp(0, 255)
        ax.imshow(curr_img.long())
        ax.scatter(xy[0], xy[1], s=50, c='white')
        joint_name = skeleton[jt_idx]['name']
        ax.set_title(joint_name)

    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    plt.close()
    im = vis_utils.fig2img(fig)
    return im


def render_2d_keypoints_im(
        im_path, input_img, gt_joint_2p5, pred_joint_2p5, hm_2d,
        joint_valid, num_frames, skeleton, hm_size, input_size):
    all_im_list = []
    batch_size = input_img.shape[0]
    num_frames = min(num_frames, batch_size)
    print('Rendering 2d keypoints (%d frames)' % (num_frames))
    for frame_idx in range(num_frames):
        im_gt = plot_2d_views(
            gt_joint_2p5, input_img, joint_valid,
            frame_idx, skeleton, hm_size, input_size)
        im_pred = plot_2d_views(
            pred_joint_2p5, input_img, joint_valid,
            frame_idx, skeleton, hm_size, input_size)
        im_2d = vis_utils.concat_pil_images([im_gt, im_pred])

        im_path_time = im_path[frame_idx][0]
        im_path_time = im_path_time.split('/')
        im_path_time = '/'.join(im_path_time[-5:-2] + im_path_time[-1:]).replace('.jpg', '')
        all_im_list.append(
            {'im': im_2d,
             'fig_name': '%s__2d_keypoints_im'%(im_path_time)})
    return all_im_list


def render_heatmap_view(
        im_path, input_img, hm_2d, pred_xy_hm,
        num_frames, skeleton, hm_size):
    all_im_list = []
    batch_size = input_img.shape[0]
    num_frames = min(num_frames, batch_size)
    print('Rendering heatmaps (%d frames)' % (num_frames))
    for frame_idx in range(num_frames):
        if frame_idx < num_frames:
            im_path_time = im_path[frame_idx][0]
            im_path_time = im_path_time.split('/')
            im_path_time = '/'.join(im_path_time[-5:-2] + im_path_time[-1:]).replace('.jpg', '')
            for v_idx in range(hm_2d.shape[1]):
                im_left = plot_hm_img(
                    hm_2d[0, v_idx], input_img[0, v_idx], pred_xy_hm[0, v_idx],
                    list(range(21, 42)), skeleton)
                im_right = plot_hm_img(
                    hm_2d[0, v_idx], input_img[0, v_idx], pred_xy_hm[0, v_idx],
                    list(range(21)), skeleton)
                all_im_list.append(
                    {'im': im_left,
                     'fig_name': '%s_hml_%d' % (im_path_time, v_idx)})
                all_im_list.append(
                    {'im': im_right,
                     'fig_name': '%s_hmr_%d' % (im_path_time, v_idx)})
    return all_im_list


def trim_white_space(fig):
    fig.gca().set_axis_off()
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    fig.margins(0, 0)
    fig.gca().xaxis.set_major_locator(plt.NullLocator())
    fig.gca().yaxis.set_major_locator(plt.NullLocator())
    return fig


def add_marker_pil(im_patch, xy):
    xy = xy.reshape(-1)
    assert len(xy) == 2
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(im_patch)
    plt.scatter(xy[0:1], xy[1:2], marker='o', s=1000, color='w', facecolors='none', linewidths=3)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    plt.axis('off')

    im_patch = vis_utils.fig2img(fig).resize((256, 256))
    plt.close()
    return im_patch


def render_target_view_proj(
        im_path, hm_2d, target_hm_2d, pts2d_proj, pts2d_gt, joint_valid,
        conf, input_img, pred_xy_hm,
        jidx, skeleton):
    sidx = 0
    num_frames = pts2d_proj.shape[0]
    im_list = []
    for time_idx in range(num_frames):
        my_path = im_path[time_idx][0]
        my_path = my_path.replace('./data/InterHand2.6M/images/', '')
        my_path = my_path.replace('.jpg', '')
        im = plot_target_view_proj(
            my_path, hm_2d, target_hm_2d, pts2d_proj, pts2d_gt, joint_valid, conf, input_img, pred_xy_hm,
            time_idx, sidx, jidx, skeleton)
        im_list.append(im)
    return im_list


def plot_target_view_proj(
        my_path, hm_2d, target_hm_2d, pts2d_proj, pts2d_gt, joint_valid, conf, input_img, pred_xy_hm,
        time_idx, sidx, jidx, skeleton):
    assert len(hm_2d.shape) == 5
    time_batch, view_batch, num_joint = hm_2d.shape[:3]
    assert num_joint == 42
    num_views = min(4, view_batch)
    view_proj_images = []
    for tidx in range(num_views):
        pts_gt = pts2d_gt[time_idx, tidx]
        valid_gt = joint_valid[time_idx, tidx]
        source_patch = input_img[time_idx, sidx]
        source_patch = torch.FloatTensor(source_patch)
        source_patch = source_patch.permute(1, 2, 0)
        source_patch = Image.fromarray((source_patch * 255).numpy().astype(np.uint8))

        target_patch = input_img[time_idx, tidx]
        target_patch = torch.FloatTensor(target_patch)
        target_patch = target_patch.permute(1, 2, 0)
        im_patch = Image.fromarray((target_patch * 255).numpy().astype(np.uint8))
        im_patch_gt = im_patch.copy()

        # source 2d keyppoint pred
        vis_keypoints(
            source_patch, pts2d_proj[time_idx, sidx, sidx],
            joint_valid[time_idx, sidx], skeleton, None)

        # target 2d keypoints projected from source
        vis_keypoints(im_patch, pts2d_proj[time_idx, sidx, tidx], valid_gt, skeleton, None)

        # gt 2d keypoints at target
        vis_keypoints(im_patch_gt, pts_gt, valid_gt, skeleton, None)

        # joint prediction at target view in len(xy) == 2
        xy = pred_xy_hm[time_idx, tidx, jidx]/64*256

        if target_hm_2d is not None:
            curr_gt_hm = target_hm_2d[time_idx, tidx, jidx].copy()
            curr_gt_hm -= curr_gt_hm.min()
            curr_gt_hm /= curr_gt_hm.max()
            curr_gt_hm *= 255
            curr_gt_hm = curr_gt_hm.astype(np.uint8)

            hm_gt = Image.fromarray(curr_gt_hm)
            hm_gt = hm_gt.resize((256, 256), Image.BILINEAR)

        curr_hm = hm_2d[time_idx, tidx, jidx].copy()

        curr_hm -= curr_hm.min()
        curr_hm /= curr_hm.max()
        curr_hm *= 255
        curr_hm = curr_hm.astype(np.uint8)

        blue_hm = np.zeros((64, 64, 3), dtype=np.uint8)
        blue_hm[:, :, 0] = curr_hm
        blue_hm[:, :, 1] = curr_hm
        hm_patch = Image.fromarray(blue_hm)
        hm_patch = hm_patch.resize((256, 256), Image.BILINEAR)

        im_blend = Image.blend(im_patch, hm_patch, 0.6)
        im_patch = add_marker_pil(im_patch, xy)

        img_cat = [
            source_patch, im_patch,
            hm_patch, im_blend, im_patch_gt]

        if target_hm_2d is not None:
            img_cat += [hm_gt]
        view_proj_images.append(
            vis_utils.concat_pil_images(img_cat))
    final_im = vis_utils.stack_pil_images(view_proj_images)
    name = skeleton[jidx]['name']
    return {'fig_name': "%s_%s_cross_proj" % (my_path, name), 'im': final_im}


def plot_src_pointclouds(
        im_path, imgs, samples_2d_gt, samples, joint_valid_unlabel,
        joint_heatmap_out, target_joint_heatmap):
    joint_valid = joint_valid_unlabel
    imgs = imgs.reshape(-1, 3, 256, 256)
    batch_size = imgs.shape[0]
    samples_2d_gt = samples_2d_gt.reshape(batch_size, 42, -1, 2)
    samples = samples.reshape(batch_size, 42, -1, 2)
    joint_valid = joint_valid.reshape(batch_size, 42)
    joint_heatmap_out = joint_heatmap_out.reshape(batch_size, 42, 64, 64, 64)
    target_joint_heatmap = target_joint_heatmap.reshape(batch_size, 42, 64, 64, 64)
    im_path = im_path.reshape(-1).tolist()

    valid_idx = 0
    out_list = []
    for idx in range(imgs.shape[0]):
        curr_img = (np.moveaxis(imgs[idx], (0), (2)) * 255).astype(np.uint8)
        gt_smpl = samples_2d_gt[idx, 4]/64.0*256
        pred_smpl = samples[idx, 4]/64*256
        curr_valid = joint_valid[idx, 4].astype(np.uint8) == 1
        if not curr_valid:
            continue
        valid_idx += 1
        if valid_idx > 5:
            break

        pred_hm = joint_heatmap_out[idx, 4].sum(axis=2).T

        gt_hm = target_joint_heatmap[idx, 4].sum(axis=2).T

        fig = plt.figure(figsize=(15, 8))
        plt.subplot(1, 4, 1)
        plt.imshow(curr_img)
        if curr_valid:
            plt.scatter(gt_smpl[:, 0], gt_smpl[:, 1], marker='x', color='w')

        plt.subplot(1, 4, 2)
        plt.imshow(curr_img)
        if curr_valid:
            plt.scatter(
                    pred_smpl[:, 0], pred_smpl[:, 1], marker='x', color='w')

        plt.subplot(1, 4, 3)
        plt.imshow(gt_hm)

        plt.subplot(1, 4, 4)
        plt.imshow(pred_hm)
        im = vis_utils.fig2img(fig)
        my_path = im_path[idx].replace(
                './data/InterHand2.6M/images/', '').replace('.jpg', '')
        im_packet = {'fig_name': "%s_src_ptcld" % (my_path), 'im': im}
        plt.close()
        out_list.append(im_packet)
    return out_list
