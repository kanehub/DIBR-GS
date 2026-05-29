import numpy as np
import cv2
import matplotlib.pyplot as plt

def rescale_depth_map(relative_depth_map, key_points, ref_depths, colmap_errors=None):
    '''
    relative_depth_map: relative depth map predicted by the model
    key_points： key pts locations in the image
    ref_depths：ref gt depth
    '''
    ref_depths = ref_depths.squeeze()
    depths = bilinear_interpolation(relative_depth_map, key_points)
    if colmap_errors is not None:
        # transfer error to weight, and normalize, then take the square root
        weight = 1. / colmap_errors
        sqrt_w = np.sqrt(weight / np.max(weight))

        depths = depths * sqrt_w
        ref_depths = ref_depths * sqrt_w

        w_min = np.percentile(sqrt_w,3)
        w_max = np.percentile(sqrt_w,97)

        mask1 = sqrt_w > w_min
        mask2 = sqrt_w < w_max

        mask = np.logical_and(mask1, mask2)

        depths = depths[mask]
        ref_depths = ref_depths[mask]

    r, b = estimate_scale_offset_lstsq(depths, ref_depths)
    rescaled_depth_map = r * relative_depth_map + b
    return rescaled_depth_map


def estimate_scale_offset_lstsq(x, y):
    """
    use numpy.linalg.lstsq for linear regression to estimate the scale and offset between x and y
    :param x: numpy array, shape (N,)
    :param y: numpy array, shape (N,)
    :return: (a, b), scale and offset
    """
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b


def bilinear_interpolation(depth_map, points):
    """
    conduct bilinear interpolation sampling on the depth map at the given points
    :param depth_map: numpy array，shape (H, W)
    :param points: numpy array，shape (N, 2) coordinates, (x, y) in each row
    """
    h, w = depth_map.shape
    sampled_values = np.zeros(len(points))

    for i, (x, y) in enumerate(points):

        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)

        # weight
        dx = x - x0
        dy = y - y0

        # for corner values
        v00 = depth_map[y0, x0]
        v10 = depth_map[y0, x1]
        v01 = depth_map[y1, x0]
        v11 = depth_map[y1, x1]

        # bilinear interpolation
        v0 = v00 * (1 - dx) + v10 * dx
        v1 = v01 * (1 - dx) + v11 * dx
        value = v0 * (1 - dy) + v1 * dy

        sampled_values[i] = value

    return sampled_values

def get_rmse(y_pred, y_true):
    squared_diff = (y_true - y_pred) ** 2
    mse = np.mean(squared_diff)
    rmse = np.sqrt(mse)
    return rmse

def visualize_depth_with_jet_colormap(depth_map, vmin=None, vmax=None, plt_show=False):
    """
    visualize depth map
    """
    # delete extreme values
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(depth_map, (1, 99))

    vmin -= 0.03
    vmax += 0.15

    norm_temp_img = np.clip((depth_map - vmin) / (vmax - vmin),0,1)
    normalized_depth = np.uint8(255 * norm_temp_img)

    # depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
    depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_TURBO)

    if plt_show:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
        # plt.imshow(depth_colormap)
        plt.title('Depth Map (Jet Colormap)')
        plt.axis('off')
        plt.show()

    return depth_colormap

def fill_holes_fast_avg(depth_map, invalid_mask, max_iter=100):
    """
    use fast average filtering iteratively to fill holes,
    """
    filled_depth = depth_map.copy()
    valid_mask = ~invalid_mask

    dist_transform = cv2.distanceTransform(valid_mask.astype(np.uint8),
                                           cv2.DIST_L2, 5)

    # for fat to near
    hole_coords = np.column_stack(np.where(invalid_mask))
    hole_distances = dist_transform[invalid_mask]

    # from border
    sort_idx = np.argsort(-hole_distances)  # 从大到小
    hole_coords = hole_coords[sort_idx]

    for iter_idx, (y, x) in enumerate(hole_coords):

        # 3x3 neighborhood
        y1, y2 = max(0, y - 1), min(depth_map.shape[0], y + 1)
        x1, x2 = max(0, x - 1), min(depth_map.shape[1], x + 1)
        neighborhood = filled_depth[y1:y2, x1:x2]

        valid_neighbors = neighborhood[neighborhood > 0]
        if len(valid_neighbors) > 0:
            filled_depth[y, x] = np.mean(valid_neighbors)
            valid_mask[y, x] = True

    return filled_depth
