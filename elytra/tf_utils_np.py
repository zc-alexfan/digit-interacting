import numpy as np

# Source: InterHand2.6M
def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord

def cam2world(cam_coord, camrot, campos):
    joint_world = (np.dot(np.linalg.inv(camrot), cam_coord.T) + campos.reshape(3, 1)).T
    return joint_world

def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord

