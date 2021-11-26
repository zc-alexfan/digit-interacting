import torch
import numpy as np
import elytra.torch_utils as torch_utils


def heatmap_argmax_3d(joint_heatmap_out, output_hm_shape):
    val_z, idx_z = torch.max(joint_heatmap_out, 2)
    val_zy, idx_zy = torch.max(val_z, 2)
    val_zyx, joint_x = torch.max(val_zy, 2)
    joint_x = joint_x[:, :, None]
    joint_y = torch.gather(idx_zy, 2, joint_x)
    joint_z = torch.gather(idx_z, 2, joint_y[:, :, :, None].repeat(
        1, 1, 1, output_hm_shape[1]))[:, :, 0, :]
    joint_z = torch.gather(joint_z, 2, joint_x)
    joint_coord_out = torch.cat((joint_x, joint_y, joint_z), 2).float()
    return joint_coord_out


def hm2xy(joint_heatmap_out, output_hm_shape, beta):
    batch_size, num_joints, hm_size = joint_heatmap_out.shape[:3]
    joint_xy = torch_utils.softargmax_kd(
        joint_heatmap_out.view(
            batch_size*num_joints, hm_size, hm_size), beta)
    xyz_idx = np.array([1, 0])  # yx to xy
    joint_xy = joint_xy.view(batch_size, num_joints, 2)
    joint_xy[:, :, :] = joint_xy[:, :, xyz_idx]
    return joint_xy


