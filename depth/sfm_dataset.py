import os
import numpy as np
from plyfile import PlyData, PlyElement

from depth.depth_utils import fill_holes_fast_avg

import cv2

from scene.colmap_loader import read_points3D_binary, read_points3D_text

class LevirSPCDataset:
    '''
    load ply
    '''
    def __init__(self, dataset_path):
        self.ply_path = os.path.join(dataset_path, 'sparse/0/points3D.ply')
        self.image_dir = os.path.join(dataset_path, 'Images')
        self.cams_dir = os.path.join(dataset_path, 'Cams')
        self.depth_dir = os.path.join(dataset_path, 'Depths')

        if not os.path.exists(self.ply_path):
            bin_path = os.path.join(dataset_path, "sparse/0/points3D.bin")
            txt_path = os.path.join(dataset_path, "sparse/0/points3D.txt")

            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            self.storePly(self.ply_path, xyz, rgb)

    def load_image_and_poses(self, img_name_list):

        self.img_attr = {}
        for img_name in img_name_list:
            self.img_attr[img_name] = {}
            image_path = os.path.join(self.image_dir, img_name)
            image = cv2.imread(image_path)
            K, w2c = self.read_camera_parameters(img_name)
            self.img_attr[img_name] = {
                'image':image,
                'K': K,
                'w2c': w2c
            }


    def load_ply(self):
        plydata = PlyData.read(self.ply_path)
        vertex = plydata['vertex']
        point_cloud = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T
        point_rgb = np.vstack((vertex['red'], vertex['green'], vertex['blue'])).T
        errors = vertex['xyzerr'] / (np.min(vertex['xyzerr'] + 1e-8)) if 'xyzerr' in vertex else None
        return point_cloud, errors

    def get_kpts_depth(self, image_name, is_visualize=False):

        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)

        point_cloud, errors = self.load_ply()
        K, w2c = self.read_camera_parameters(image_name)
        key_points, depth = self.projected2d(point_cloud, K, w2c, image)

        if is_visualize:
            self.visualize_points(image, image_name, key_points)

        return key_points, depth, errors

    def visualize_points(self, image, image_name, n_key_points, idx=150):
        result_img = image.copy()
        key_point = n_key_points[idx].astype(np.int32)
        cv2.circle(result_img, tuple(key_point), 3, (0, 0, 255), 1)
        # result_img = cv2.resize(result_img, dsize=None, fy=0.5, fx=0.5)
        cv2.imshow(f'{image_name}', result_img)
        cv2.waitKey(0)

    def read_camera_parameters(self, image_name):
        image_idx = image_name.split('.')[0]
        filename = os.path.join(self.cams_dir, f'{image_idx}.txt')
        with open(filename, "r") as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        extrinsic = np.fromstring(",".join(lines[1:5]), dtype=np.float32, sep=",").reshape(4, 4)
        intrinsic = np.fromstring(",".join(lines[7:10]), dtype=np.float32, sep=",").reshape(3, 3)
        depth_min, depth_max = [float(item) for item in lines[11].split(",")]
        return intrinsic, extrinsic

    def projected2d(self, points, K, w2c, image, check_valid=True):
        '''
        # project 3d pts to 2d
        '''
        points_world = points[..., np.newaxis]  # [N, 3, 1]
        R = w2c[:3, :3][np.newaxis, ...]  # [1, 3, 3]
        T = w2c[:3, 3:][np.newaxis, ...]  # [1, 3, 1]
        points_camera = np.matmul(R, points_world) + T  # [N, 3, 1]

        depth = points_camera[..., 2, :]  # [N, 1]
        # to pixel coordinate
        K = K[np.newaxis, ...]  # [1, 3, 3]
        points_uv = np.matmul(K, points_camera)  # [N, 3, 1]

        # normalization
        points_uv = points_uv.squeeze()
        points_uv = points_uv[..., :2] / points_uv[..., 2:]

        # check if the projected pts are within the image boundaries
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

    def reproject(self, depth_map, K, w2c):
        h, w = depth_map.shape
        u = np.arange(w)
        v = np.arange(h)
        uu, vv = np.meshgrid(u, v)
        pixel_coords = np.stack([uu, vv, np.ones_like(uu)], axis=-1) # [H,W,3]
        inv_k = np.linalg.inv(K) # [3,3]

        # for boardcasting, do not use inv_k @ pixel_coords, because the mismatch of dimension
        # use pixel_coords @ inv_k^T
        # equal to  P^T @ (inv_k)^T = (inv_k @ P)^T
        camera_coords = np.dot(pixel_coords, inv_k.T)
        points_3d_camera = camera_coords * depth_map[:, :, np.newaxis]

        points_3d_camera = points_3d_camera.reshape(-1,3)

        # camera to world
        c2w = np.linalg.inv(w2c)

        n_points = points_3d_camera.shape[0]
        points_homogeneous = np.column_stack([points_3d_camera, np.ones(n_points)])

        points_world_homogeneous = points_homogeneous @ c2w.T
        points_3d_world = points_world_homogeneous[:, :3]

        return points_3d_world

    def get_pcd_from_depth(self, image_name, depth_map):
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1,3)
        K, w2c = self.read_camera_parameters(image_name)
        points_3d = self.reproject(depth_map, K, w2c)

        return points_3d, img_rgb

    def depth_warp(self, image_name, points_3d):
        image = self.img_attr[image_name]['image']
        height, width = image.shape[:2]
        K = self.img_attr[image_name]['K']
        w2c = self.img_attr[image_name]['w2c']
        points_uv, src_depth = self.projected2d(points_3d, K, w2c, image, check_valid=True)
        x_res = [uv[0] for uv in points_uv]
        y_res = [uv[1] for uv in points_uv]
        y_res = np.clip(np.floor(y_res), 0, height - 1)
        x_res = np.clip(np.floor(x_res), 0, width - 1)
        y_res = y_res.astype(np.int64)
        x_res = x_res.astype(np.int64)
        new_depth = np.zeros((height, width), dtype=np.float32)

        new_depth[y_res, x_res] = src_depth[:,0]

        # fill the holes
        invalid_mask = new_depth<1
        fill_depth = fill_holes_fast_avg(new_depth, invalid_mask)

        is_vis_warp = False
        if is_vis_warp:
            warp_mask = fill_depth > 0
            warp_mask = warp_mask.astype(np.uint8) * 255
            cv2.imshow(f'{image_name} mask', warp_mask)
            cv2.waitKey(0)

        return fill_depth



    def depth_refine(self, src_depth, depth_arr):
        # keep the farthest depth value for each pixel
        depth_arr.append(src_depth)
        n_depths = np.stack(depth_arr, axis=0)
        final_depth = np.max(n_depths, axis=0)
        return final_depth


    def storePly(self, path, xyz, rgb):
        # Define the dtype for the structured array
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

        normals = np.zeros_like(xyz)

        elements = np.empty(xyz.shape[0], dtype=dtype)
        attributes = np.concatenate((xyz, normals, rgb), axis=1)
        elements[:] = list(map(tuple, attributes))

        # Create the PlyData object and write to file
        vertex_element = PlyElement.describe(elements, 'vertex')
        ply_data = PlyData([vertex_element])
        ply_data.write(path)
