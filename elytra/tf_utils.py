import torch

def pixel2cam_torch(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = torch.stack((x, y, z), dim=1)
    return cam_coord

def pixel2cam_batch(pixel_coord, f, c):
    batch_size = pixel_coord.shape[0]
    assert pixel_coord.shape[2] == 3
    assert f.shape == (batch_size, 2)
    assert c.shape == (batch_size, 2)

    x = ((pixel_coord[:, :, 0] - c[:, 0][:, None])/f[:, 0][:, None]) * pixel_coord[:, :, 2]
    y = ((pixel_coord[:, :, 1] - c[:, 1][:, None])/f[:, 1][:, None]) * pixel_coord[:, :, 2]
    z = pixel_coord[:, :, 2]
    cam_coord = torch.stack((x, y, z), dim=2)
    return cam_coord


def cam2world_batch(joint_xyz, camrot, campos):
    bs = joint_xyz.shape[0]
    assert camrot.shape == (bs, 3, 3)
    assert campos.shape == (bs, 3)
    camrot_inv = torch.inverse(camrot)
    joint_world = (torch.bmm(
        camrot_inv, joint_xyz.permute(0, 2, 1)) + campos[:, :, None]).permute(0, 2, 1)
    assert joint_world.shape[0] == bs
    assert joint_world.shape[2] == 3
    return joint_world


def cam2pixel_batch(cam_coord, f, c, eps=1e-8):
    """
    Projects points in camera coord to pixel space given focal and princpt.
    cam_coord: (batch, num_joints, 3)
    f: (batch, 2)
    c: (batch, 2)
    Outputs: (batch, num_joints, 2)
    """
    assert isinstance(cam_coord, torch.Tensor)
    batch_size = cam_coord.shape[0]
    assert cam_coord.shape[2] == 3
    assert f.shape == (batch_size, 2)
    assert c.shape == (batch_size, 2)
    x = cam_coord[:, :, 0] / (cam_coord[:, :, 2] + eps) * f[:, 0:1] + c[:, 0:1]
    y = cam_coord[:, :, 1] / (cam_coord[:, :, 2] + eps) * f[:, 1:2] + c[:, 1:2]
    pix = torch.stack((x, y), dim=2)
    assert pix.shape[0] == batch_size
    assert pix.shape[2] == 2
    return pix


def world2cam_batch(world_coord, R, T):
    """
    Convert from world to cam coordinates
    world_coord: (batch, num_joints, 3)
    R: (batch, 3, 3)
    T: (batch, 3)
    Outputs: (batch, num_joints, 3)
    """
    batch_size, joint_num, _ = world_coord.shape
    assert R.shape == (batch_size, 3, 3)
    assert T.shape == (batch_size, 3)
    R = R.permute(0, 2, 1)
    cam_coord = torch.bmm(world_coord - T[:, None, :], R)
    assert cam_coord.shape == world_coord.shape
    return cam_coord


