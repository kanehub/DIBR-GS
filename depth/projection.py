import numpy as np


def projected2d(points, K, w2c, image, check_valid=True):
    '''
    # project 3d to 2d
    '''
    points_world = points[..., np.newaxis]  # [N, 3, 1]
    R = w2c[:3, :3][np.newaxis, ...]  # [1, 3, 3]
    T = w2c[:3, 3:][np.newaxis, ...]  # [1, 3, 1]
    points_camera = np.matmul(R, points_world) + T  # [N, 3, 1]

    depth = points_camera[..., 2, :]  # [N, 1]
    # to pixel coordinate
    K = K[np.newaxis, ...]  # [1, 3, 3]
    points_uv = np.matmul(K, points_camera)  # [N, 3, 1]

    # normalize
    points_uv = points_uv.squeeze()
    points_uv = points_uv[..., :2] / points_uv[..., 2:]

    # check if the projected points are valid (in the image)
    if check_valid:
        H, W = image.shape[:2]
        valid_mask_x = np.logical_and(points_uv[..., 0] > 0, points_uv[..., 0] < W)
        valid_mask_y = np.logical_and(points_uv[..., 1] > 0, points_uv[..., 1] < H)
        valid_mask = np.logical_and(valid_mask_x, valid_mask_y)
        valid_mask = valid_mask.squeeze()

        points_uv = points_uv[valid_mask]
        depth = depth[valid_mask]

    depth = np.abs(depth)

    return points_uv, depth