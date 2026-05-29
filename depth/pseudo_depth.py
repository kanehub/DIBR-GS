import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from plyfile import PlyData
from tqdm import tqdm

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth.projection import projected2d
from depth.depth_utils import visualize_depth_with_jet_colormap

class DepthWarper:
    def __init__(self, args):
        dataset_path = os.path.join(args.dataset_dir, args.scene_id)
        self.ply_path = os.path.join(dataset_path, r'sparse/mono_depth/view_no_ovlp.ply')
        self.dataset_path = dataset_path
        self.cams_dir = os.path.join(dataset_path, 'Cams')
        self.img_dir = os.path.join(dataset_path, 'Images')
        self.gt_depth_dir = os.path.join(dataset_path, 'Depths')

        # self.train_set_names = ['001.png', '008.png', '015.png']
        self.train_set_names = [f"{idx:03d}.png" for idx in args.train_list]
        img_names = os.listdir(self.img_dir)
        self.test_cam_names = [name for name in img_names if name not in self.train_set_names]
        self.pseudo_dir = os.path.join(dataset_path, 'pseudo_images')

    def run(self, vis_error=False):
        mono_pcd, rgb_pcd = self.load_ply(self.ply_path)
        if os.path.exists(self.pseudo_dir) is False:
            os.makedirs(self.pseudo_dir)

        for img_name in tqdm(self.test_cam_names, desc="Pseudo View generating", ncols=80):

            image = self.read_image(img_name)
            K, w2c = self.read_camera_parameters(img_name)
            pseudo_depth_ori = forward_splat(image, mono_pcd, K, w2c)
            pseudo_depth = fill_depth_nearest(pseudo_depth_ori)

            if vis_error:
                self.vis_depth_error(img_name, pseudo_depth)

            seq = self.get_near_squence(img_name)
            tgt_list = []
            for train_idx in seq:

                src_img_name = self.train_set_names[train_idx]
                src_image = self.read_image(src_img_name)
                _, src_w2c = self.read_camera_parameters(src_img_name)

                tgt_img, tgt_mask = backward_warp(src_image, pseudo_depth, K, src_w2c, w2c)
                tgt_list.append((tgt_img, tgt_mask))

            merged_img, merge_mask = self.merge_imgs(tgt_list)

            # padding border
            src_img_name = self.train_set_names[seq[0]]
            src_image = self.read_image(src_img_name)
            _, src_w2c = self.read_camera_parameters(src_img_name)
            padding_img, _ = backward_warp(src_image, pseudo_depth, K, src_w2c, w2c, padding=True)

            padding_mask = ~merge_mask
            merged_img[padding_mask] = padding_img[padding_mask]

            # save
            cv2.imwrite(os.path.join(self.pseudo_dir, img_name), merged_img)
            print(f"Save {img_name} pseudo image.")

    def read_image(self, img_name):
        image_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(image_path)
        return image

    def read_depth(self, img_name):
        gt_depth_name = f"{img_name.split('.')[0]}.tiff"
        gt_depth_path = os.path.join(self.dataset_path, 'Depths', gt_depth_name)
        gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)

        return gt_depth
    def read_camera_parameters(self, image_name):
        image_idx = image_name.split('.')[0]
        filename = os.path.join(self.cams_dir, f'{image_idx}.txt')
        with open(filename, "r") as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        extrinsic = np.fromstring(",".join(lines[1:5]), dtype=np.float32, sep=",").reshape(4, 4)
        intrinsic = np.fromstring(",".join(lines[7:10]), dtype=np.float32, sep=",").reshape(3, 3)
        return intrinsic, extrinsic

    def load_ply(self, ply_path):
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        point_cloud = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T
        point_rgb = np.vstack((vertex['red'], vertex['green'], vertex['blue'])).T
        return point_cloud, point_rgb

    def vis_depth_error(self, img_name, pseudo_depth):
        gt_depth = self.read_depth(img_name)
        valid_mask = (pseudo_depth > 0)
        error_map = np.abs(pseudo_depth - gt_depth) * valid_mask
        rmse = np.sqrt(np.sum(error_map ** 2) / np.sum(valid_mask))
        print(f"{img_name}: RMSE = {rmse:.4f}")
        error_map = np.abs(pseudo_depth - gt_depth)
        visualize_depth_with_jet_colormap(error_map, vmin=0, vmax=10)

    def get_near_squence(self, img_name):
        img_idx = img_name.split('.')[0]
        train_indices = [int(name.split('.')[0]) for name in self.train_set_names]
        tgt_idx = int(img_idx)
        dists = [min(abs(tgt_idx - idx), abs(tgt_idx - (idx+20))) for idx in train_indices]
        dists = np.array(dists)
        seq = np.argsort(dists)

        return seq

    def merge_imgs(self, img_list):
        img, mask = img_list[0]
        merge_mask = mask.copy()
        result = img.copy()

        idx = 2
        for img_tuple in img_list[1:]:
            next_img, next_mask = img_tuple
            new_area = next_mask & ~merge_mask
            result[new_area] = next_img[new_area]
            merge_mask |= new_area
            # self.plot_img(result, f'{idx}')
            idx += 1

        return result, merge_mask

def forward_splat(image, points_3d, K, w2c):
    '''
    generate depth map by forward splatting the point cloud
    '''
    height, width = image.shape[:2]
    points_uv, src_depth = projected2d(points_3d, K, w2c, image, check_valid=True)
    depth = src_depth[:, 0].astype(np.float32)

    x = np.floor([uv[0] for uv in points_uv]).astype(np.int64)
    y = np.floor([uv[1] for uv in points_uv]).astype(np.int64)

    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)

    new_depth = np.zeros((height, width), dtype=np.float32)

    # use the farthest (maximum) depth if multiple points fall on the same pixel
    np.maximum.at(new_depth, (y, x), depth)

    return new_depth
def forward_splat_rgb(image, points_3d, points_rgb, K, w2c):
    H, W = image.shape[:2]
    points_uv, src_depth = projected2d(
        points_3d, K, w2c, image, check_valid=True
    )
    depth = src_depth[:, 0].astype(np.float32)

    x = np.floor([uv[0] for uv in points_uv]).astype(np.int64)
    y = np.floor([uv[1] for uv in points_uv]).astype(np.int64)
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)

    # -------- 1. initialization --------
    depth_map = np.full((H, W), 0, dtype=np.float32)
    rgb_map = np.zeros((H, W, 3), dtype=np.int64)

    # -------- 2. Traverse by points--------
    for i in range(len(depth)):
        xi, yi = x[i], y[i]
        di = depth[i]

        if di > depth_map[yi, xi]:
            depth_map[yi, xi] = di
            rgb_map[yi, xi] = points_rgb[i]

    return depth_map, rgb_map


def backward_warp(src_img, tgt_depth, K, src_w2c, tgt_w2c, padding=False):
    """
    Args:
        src_img: (Hs, Ws, C) uint8/float32
        tgt_depth: (H, W) float32
        K: (3, 3) float32
        src_w2c: (4, 4) float32  world->src_cam
        tgt_w2c: (4, 4) float32  world->tgt_cam

    Returns:
        tgt_img: (H, W, C) same dtype as src_img (by cv2.remap)
        valid_mask: (H, W) bool
    """
    assert K.shape == (3, 3)
    assert src_w2c.shape == (4, 4) and tgt_w2c.shape == (4, 4)

    H, W = tgt_depth.shape[:2]
    Hs, Ws = src_img.shape[:2]
    dtype_img = src_img.dtype

    # --- 1) build target pixel grid (u,v,1) ---
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H,W)

    ones = np.ones_like(uu, dtype=np.float32)
    pix = np.stack([uu, vv, ones], axis=-1).reshape(-1, 3).T  # (3, HW)

    # --- 2) unproject to target camera 3D: X_t = inv(K) * pix * depth ---
    K_inv = np.linalg.inv(K).astype(np.float32)
    depth = tgt_depth.reshape(-1).astype(np.float32)  # (HW,)
    X_t = (K_inv @ pix) * depth[None, :]             # (3, HW)

    # valid depth mask (depth > 0)
    valid = depth > 0

    # --- 3) transform target cam -> world -> src cam ---
    tgt_c2w = np.linalg.inv(tgt_w2c).astype(np.float32)
    src_w2c = src_w2c.astype(np.float32)

    X_t_h = np.vstack([X_t, np.ones((1, X_t.shape[1]), dtype=np.float32)])  # (4, HW)
    X_w_h = tgt_c2w @ X_t_h                                                 # (4, HW)
    X_s_h = src_w2c @ X_w_h                                                 # (4, HW)
    X_s = X_s_h[:3, :]                                                      # (3, HW)

    zs = X_s[2, :]
    valid &= zs > 1e-6  # in front of src camera

    # --- 4) project to source pixels: x = K * X_s ---
    x = K.astype(np.float32) @ X_s  # (3, HW)
    xs = x[0, :] / x[2, :]
    ys = x[1, :] / x[2, :]

    # --- 5) sample src_img at (xs, ys) to form target image ---
    map_x = xs.reshape(H, W).astype(np.float32)
    map_y = ys.reshape(H, W).astype(np.float32)

    if padding:

        tgt_img = cv2.remap(
            src_img,
            map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(dtype_img)

        valid_mask = valid.reshape(H, W)
    else:
        # image bounds
        valid &= (xs >= 0) & (xs <= Ws - 1) & (ys >= 0) & (ys <= Hs - 1)

        # For invalid points, give an out-of-bounds coordinate to avoid remap mis-sampling.
        map_x[~valid.reshape(H, W)] = -1.0
        map_y[~valid.reshape(H, W)] = -1.0

        tgt_img = cv2.remap(
            src_img,
            map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        ).astype(dtype_img)

        valid_mask = valid.reshape(H, W)

    return tgt_img, valid_mask


def fill_depth_nearest(depth, invalid=0):
    d = depth.astype(np.float32).copy()
    mask = (d == invalid) | ~np.isfinite(d)
    #  8-bit mask：1 needs to be filled
    m8 = mask.astype(np.uint8) * 255
    filled = cv2.inpaint(d, m8, 3, cv2.INPAINT_TELEA)
    return filled

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", help="path to dataset", type=str, default="../data/LEVIR_NVS")
    parser.add_argument("-s", "--scene_id", help="scene id",type=str, default="scene_000")
    parser.add_argument("-t", "--train_list", type=int, default=[1, 8, 15], nargs='*', help='training_idx')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    depth_warper = DepthWarper(args)
    depth_warper.run(vis_error=False)
