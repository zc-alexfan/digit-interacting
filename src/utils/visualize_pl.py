from PIL import Image
import numpy as np
import elytra.vis_utils as vis_utils
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import src.utils.vis as vis
from src.utils.preprocessing import load_skeleton
from src.utils.config import cfg as args
import os.path as op

skeleton = load_skeleton(
    op.join(args.anno_path, 'skeleton.txt'), 21*2)


def prepare_data(vis_dict):
    img = vis_dict['input_img']
    gt_joint_2p5 = vis_dict['gt_joint_2p5']
    pred_joint_2p5 = vis_dict['pred_joint_2p5']
    pred_joint_cam = vis_dict['pred_cam']
    gt_joint_cam = vis_dict['gt_cam']
    im_path = vis_dict['im_path']
    joint_valid = vis_dict['joint_valid']
    joint_hm_2d = vis_dict['hm_2d']
    gt_joint_2d = gt_joint_2p5[:, :, :2]/64.0*256
    pred_joint_2d = pred_joint_2p5[:, :, :2]/64.0*256

    raw_im_list = []
    for im_np in img:
        im_np = (im_np/im_np.max()*255).astype(np.uint8)
        im = Image.fromarray(np.moveaxis(im_np, (0, 1, 2), (2, 0, 1)))
        raw_im_list.append(im)
    del img

    im_path = [
            ip.replace(
                './data/InterHand2.6M/images/', ''
            ).replace('.jpg', '') for ip in im_path]

    plt_dict = {}
    plt_dict['raw_im_list'] = raw_im_list
    plt_dict['gt_2d'] = gt_joint_2d
    plt_dict['pred_2d'] = pred_joint_2d
    plt_dict['gt_2p5'] = gt_joint_2p5
    plt_dict['pred_2p5'] = pred_joint_2p5
    plt_dict['pred_cam'] = pred_joint_cam
    plt_dict['gt_cam'] = gt_joint_cam
    plt_dict['joint_valid'] = joint_valid
    plt_dict['im_path'] = im_path
    plt_dict['hm_2d'] = joint_hm_2d
    plt_dict['hm_norm'] = vis_dict['hm_norm']
    plt_dict['segm_l_mask'] = vis_dict['segm_l_mask']
    plt_dict['segm_r_mask'] = vis_dict['segm_r_mask']
    plt_dict['segm_valid'] = vis_dict['segm_valid']
    plt_dict['segm_target_256'] = vis_dict['segm_target_256']
    return plt_dict


def plot_views_3d(
        im, gt_joint_2d, pred_joint_2d,
        gt_joint_2p5, pred_joint_2p5,
        joint_valid, skeleton, prefix, postfix):

    """
    Visualize 2d and 3d skeleton for ONE example.
    """
    assert isinstance(im, Image.Image)
    assert isinstance(gt_joint_2d, np.ndarray)
    assert isinstance(pred_joint_2d, np.ndarray)
    assert isinstance(gt_joint_2p5, np.ndarray)
    assert isinstance(pred_joint_2p5, np.ndarray)
    assert isinstance(joint_valid, np.ndarray)
    assert gt_joint_2d.shape == (42, 2)
    assert gt_joint_2p5.shape == (42, 3)
    assert gt_joint_2p5.shape == pred_joint_2p5.shape
    assert gt_joint_2d.shape == pred_joint_2d.shape

    im_list = vis.generate_vis_images(
            im, gt_joint_2d, pred_joint_2d, gt_joint_2p5, pred_joint_2p5,
            joint_valid, skeleton)
    im_2p5 = vis_utils.concat_pil_images(im_list)
    return {'im': im_2p5, 'fig_name': prefix + postfix}

def vis_amodal_rgb(
        curr_l_mask, curr_r_mask,
        curr_segm_target_l, curr_segm_target_r, curr_img):

    curr_r_mask_clone = curr_r_mask.clone()
    curr_r_mask_clone = curr_r_mask_clone.float()
    curr_r_mask_clone[curr_r_mask_clone == 0] = np.nan
    diff_r = (curr_r_mask != curr_segm_target_r).long()

    curr_l_mask_clone = curr_l_mask.clone()
    curr_l_mask_clone = curr_l_mask_clone.float()
    curr_l_mask_clone[curr_l_mask_clone == 0] = np.nan
    diff_l = (curr_l_mask != curr_segm_target_l).long()

    fig, ax = plt.subplots(2, 5, figsize=(15, 8))
    ax[0, 0].imshow(curr_img)
    ax[0, 0].imshow(curr_r_mask_clone, alpha=0.8)
    ax[0, 1].imshow(curr_r_mask)
    ax[0, 2].imshow(curr_segm_target_r)
    ax[0, 3].imshow(diff_r)
    ax[0, 4].imshow(curr_img)

    ax[1, 0].imshow(curr_img)
    ax[1, 0].imshow(curr_l_mask_clone, alpha=0.8)
    ax[1, 1].imshow(curr_l_mask)
    ax[1, 2].imshow(curr_segm_target_l)
    ax[1, 3].imshow(diff_l)
    ax[1, 4].imshow(curr_img)
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.close()
    im = vis_utils.fig2img(fig)
    return im


def plt_img_hm_segm_kp(
        titles, curr_hm_norm, curr_pred_joint_2p5, curr_gt_joint_2p5,
        curr_img, curr_segm_mask, curr_joint_valid, joint_offset):
    jidx_list = list(np.random.permutation(21)[:5])
    fig, ax = plt.subplots(3, len(jidx_list), figsize=(20, 10))
    for sample_idx in range(len(jidx_list)):
        jidx = jidx_list[sample_idx] + joint_offset
        curr_hm_norm_jt = curr_hm_norm[jidx]
        curr_pred_joint = curr_pred_joint_2p5[jidx]
        curr_gt_joint = curr_gt_joint_2p5[jidx]

        ax[0, sample_idx].imshow(curr_img)
        ax[0, sample_idx].imshow(curr_hm_norm_jt, alpha=0.8, cmap='gnuplot')
        ax[0, sample_idx].set_title(titles[jidx])

        ax[1, sample_idx].imshow(curr_img)
        ax[1, sample_idx].imshow(curr_segm_mask, alpha=0.4)
        ax[1, sample_idx].scatter(
                curr_pred_joint[0], curr_pred_joint[1],
                marker='x', color='w', s=80)
        if int(curr_joint_valid[sample_idx]) == 1:
            ax[1, sample_idx].scatter(
                    curr_gt_joint[0], curr_gt_joint[1],
                    marker='x', color='y', s=80)

        ax[2, sample_idx].imshow(curr_segm_mask)
        ax[2, sample_idx].scatter(
                curr_pred_joint[0], curr_pred_joint[1], marker='x', color='w', s=80)
        if int(curr_joint_valid[sample_idx]) == 1:
            ax[2, sample_idx].scatter(
                    curr_gt_joint[0], curr_gt_joint[1], marker='x', color='y', s=80)

    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.close()
    im = vis_utils.fig2img(fig)
    return im


def visualize_all(
        vis_dict, max_examples, postfix, no_tqdm):

    # prepare plotting data
    plt_dict = prepare_data(vis_dict)
    del vis_dict

    titles = [skl['name'] for skl in skeleton]
    im_list = []
    max_examples = min(max_examples, len(plt_dict['im_path']))
    myrange = tqdm(range(max_examples)) if not no_tqdm else range(max_examples)
    for example_idx in myrange:
        curr_path = plt_dict['im_path'][example_idx]
        curr_gt_2d = plt_dict['gt_2d'][example_idx]
        curr_pred_2d = plt_dict['pred_2d'][example_idx]
        curr_gt_2p5 = plt_dict['gt_2p5'][example_idx]
        curr_pred_2p5 = plt_dict['pred_2p5'][example_idx]
        curr_joint_valid = plt_dict['joint_valid'][example_idx]
        curr_im = plt_dict['raw_im_list'][example_idx]
        curr_pred_cam = plt_dict['pred_cam'][example_idx]
        curr_gt_cam = plt_dict['gt_cam'][example_idx]
        curr_hm_norm = plt_dict['hm_norm'][example_idx]
        curr_segm_valid = plt_dict['segm_valid'][example_idx]

        curr_r_mask = torch.LongTensor(plt_dict['segm_r_mask'][example_idx])
        curr_l_mask = torch.LongTensor(plt_dict['segm_l_mask'][example_idx])
        curr_segm_target = torch.LongTensor(plt_dict['segm_target_256'][example_idx])
        curr_segm_target_l = torch.LongTensor(curr_segm_target[1])
        curr_segm_target_r = torch.LongTensor(curr_segm_target[2])

        # right hand
        curr_hm_norm_256 = F.interpolate(
                torch.FloatTensor(
                    curr_hm_norm[None, :, :, :]), 256).numpy().squeeze()

        im = plt_img_hm_segm_kp(
                titles, curr_hm_norm_256, curr_pred_2d, curr_gt_2d,
                curr_im, curr_r_mask, curr_joint_valid,  joint_offset=0)
        im_list.append({'im': im, 'fig_name': curr_path + '_hm_segm_kp_r'})

        im = plt_img_hm_segm_kp(
                titles, curr_hm_norm_256, curr_pred_2d, curr_gt_2d,
                curr_im, curr_l_mask, curr_joint_valid,  joint_offset=21)
        im_list.append({'im': im, 'fig_name': curr_path + '_hm_segm_kp_l'})

        im = vis_amodal_rgb(
            curr_l_mask, curr_r_mask,
            curr_segm_target_l, curr_segm_target_r, curr_im)
        im_list.append(
                {'im': im,
                 'fig_name': curr_path + '_segm_rgb_%d' % (curr_segm_valid)})

        im = plot_views_3d(
            curr_im, curr_gt_2d, curr_pred_2d, curr_gt_2p5, curr_pred_2p5,
            curr_joint_valid, skeleton,
            prefix=curr_path, postfix='_2d_2p5d_views'+postfix)
        im_list.append(im)

        im = plot_views_3d(
            curr_im, curr_gt_2d, curr_pred_2d, curr_gt_cam, curr_pred_cam,
            curr_joint_valid, skeleton,
            prefix=curr_path, postfix='_2d_cam_views'+postfix)
        im_list.append(im)
    return im_list
