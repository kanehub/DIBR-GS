import os
from PIL import Image
import cv2
import numpy as np
import argparse
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth import DepthEstimator, LevirSPCDataset
from depth.refine_point_cloud import refine_ply_with_dip
from depth.depth_utils import rescale_depth_map, get_rmse, visualize_depth_with_jet_colormap
def main(args):
    dataset_path = os.path.join(args.dataset_path, args.scene_id)
    spc_dataset = LevirSPCDataset(dataset_path)
    dpt_model = DepthEstimator()
    transfer_type = 'inverse'  # inverse or negative
    # img_name_list = ["001.png", "008.png", "015.png"]
    img_name_list = [f"{idx:03d}.png" for idx in args.train_list]

    points_arr = []
    imgs_arr = []
    depths_arr = []
    for img_name in img_name_list:
        key_points, ref_depths, _ = spc_dataset.get_kpts_depth(img_name)
        img_path = os.path.join(dataset_path, 'Images', img_name)

        image = Image.open(img_path)
        predicted_disparity = dpt_model.predict(image, mode='test')
        if transfer_type == 'inverse':
            predicted_depth = 1. / (predicted_disparity + 1.0)
        else:
            predicted_depth = -predicted_disparity

        rescaled_depth = rescale_depth_map(predicted_depth, key_points, ref_depths)
        depths_arr.append(rescaled_depth)

        is_vis_error = False
        if is_vis_error:
            gt_img_name = f"{img_name.split('.')[0]}.tiff"
            gt_depth_path = os.path.join(dataset_path, 'Depths', gt_img_name)
            gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
            rmse = get_rmse(rescaled_depth, gt_depth)
            print(f"{img_name}: rmse is ", rmse)
            error_map = np.abs(rescaled_depth - gt_depth)
            visualize_depth_with_jet_colormap(error_map, vmin=0, vmax=10)

        points_3d, img_rgb = spc_dataset.get_pcd_from_depth(img_name, rescaled_depth)

        points_arr.append(points_3d)
        imgs_arr.append(img_rgb)

    # concat
    n_points_3d = np.concatenate(points_arr, axis=0)
    n_img_rgb = np.concatenate(imgs_arr, axis=0)

    output_ply_dir = os.path.join(args.output_path, args.scene_id)
    if os.path.exists(output_ply_dir) is False:
        os.makedirs(output_ply_dir)
    output_ply_path = os.path.join(output_ply_dir, 'monodepth.ply')
    spc_dataset.storePly(output_ply_path, n_points_3d, n_img_rgb)
    print("Saved mono depth point cloud to ", output_ply_path)

    if args.refine_ply:
        refine_ply_path = os.path.join(output_ply_dir, 'no_ovlp.ply')
        refine_ply_with_dip(output_ply_path, refine_ply_path)
        if args.mv_to_dataset:
            print("Moving refined ply to dataset directory")
            tgt_ply_dir = os.path.join(dataset_path, "sparse", "mono_depth")
            if os.path.exists(tgt_ply_dir) is False:
                os.makedirs(tgt_ply_dir)
            tgt_ply_path = os.path.join(tgt_ply_dir, 'view_no_ovlp.ply')
            shutil.copyfile(refine_ply_path, tgt_ply_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", help="path to dataset", type=str, default="../data/LEVIR_NVS")
    parser.add_argument("-s", "--scene_id", help="scene id",type=str, default="scene_000")
    parser.add_argument("-t", "--train_list", type=int, default=[1, 8, 15], nargs='*', help='training_idx')
    parser.add_argument("-o", "--output_path", type=str, default="./outputs", help='output dataset path')
    parser.add_argument("--mv_to_dataset", action="store_true") # whether move sfm files to dataset
    parser.add_argument("--refine_ply", action="store_true") # whether to refine_ply point cloud using dip test

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
