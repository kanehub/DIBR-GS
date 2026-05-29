import numpy as np
from plyfile import PlyData
import diptest
import matplotlib.pyplot as plt
from scene import storePly

def fetch_ply(ply_path):
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    point_cloud = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T
    point_rgb = np.vstack((vertex['red'], vertex['green'], vertex['blue'])).T
    errors = vertex['xyzerr'] / (np.min(vertex['xyzerr'] + 1e-8)) if 'xyzerr' in vertex else None
    return point_cloud, point_rgb, errors

def refine_ply_with_dip(ply_path, output_path):
    pts, rgbs, _ = fetch_ply(ply_path)
    grid_dict, grid_info = grid_partition_with_indices(pts, 10)
    # print_grid_statistics(grid_dict, grid_info)
    delete_overlap_idx = []
    print("Start refining point cloud...")
    for k,v in grid_dict.items():
        if len(v) > 1000:
            # get pts in grid
            pts_in_grid = pts[v]
            z_values = pts_in_grid[:,-1]
            dip, pval = diptest.diptest(z_values)
            if pval<0.05:
                # significant multimodality, likely to have vertical overlaps
                # get overlap ratio
                over_lap_idx, ratio, detele_idx = find_vertical_overlaps_kdtree(pts_in_grid)
                if ratio > 0.3:
                    detele_idx = list(set(detele_idx))
                    true_detele_idx = np.array(v, dtype=np.int32)[detele_idx]
                    delete_overlap_idx.extend(true_detele_idx.tolist())

    new_pts = np.delete(pts,delete_overlap_idx, axis=0)
    new_rgbs = np.delete(rgbs,delete_overlap_idx, axis=0)

    storePly(output_path, new_pts, new_rgbs)
    print("Refining completed. Output saved to ", output_path)

def find_vertical_overlaps_kdtree(points, xy_tolerance=0.1, min_z_diff=1):
    from scipy.spatial import cKDTree

    xy_points = points[:, :2]

    # create KD tree
    tree = cKDTree(xy_points)

    # find neighbor
    pairs = tree.query_pairs(xy_tolerance, output_type='ndarray')
    cnt = 0
    over_lap_idx = []
    detele_idx = []
    for i, j in pairs:
        z_diff = abs(points[i, 2] - points[j, 2])
        if z_diff > min_z_diff:
            cnt += 1
            over_lap_idx.append(i)
            over_lap_idx.append(j)
            if points[i, 2] > points[j, 2]:
                detele_idx.append(i)
            else:
                detele_idx.append(j)

    ratio = cnt / (2 * len(pairs)) if len(pairs) > 0 else 0

    return over_lap_idx, ratio, detele_idx

def grid_partition_with_indices(points, grid_num):
    """
    Grid partitioning with indices

    Parameters:
    points: numpy array, (N, 3)
    grid_size: float, the size of each grid cell

    Returns:
    grid_dict: dict, key is grid index (i, j), value is the original indices of points in that grid
    grid_info: dict, grid information dictionary
    """

    x_coords = points[:, 0]
    y_coords = points[:, 1]

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    max_length = max((x_max-x_min),(y_max-y_min))
    grid_size = max_length // grid_num

    x_cells = int(np.ceil((x_max - x_min) / grid_size))
    y_cells = int(np.ceil((y_max - y_min) / grid_size))

    grid_dict = {}

    for point_idx in range(len(points)):
        x, y = points[point_idx, 0], points[point_idx, 1]

        i = int((x - x_min) // grid_size)
        j = int((y - y_min) // grid_size)

        i = max(0, min(i, x_cells - 1))
        j = max(0, min(j, y_cells - 1))

        if (i, j) not in grid_dict:
            grid_dict[(i, j)] = []

        grid_dict[(i, j)].append(point_idx)

    grid_info = {
        'grid_size': grid_size,
        'x_range': (x_min, x_max),
        'y_range': (y_min, y_max),
        'grid_shape': (x_cells, y_cells),
        'total_points': len(points)
    }

    return grid_dict, grid_info