import argparse
import os
import shutil
from datetime import datetime

import pycolmap
import subprocess

from levir2colmap import generate_init_txt
from read_write_model import read_cameras_binary, read_images_binary, read_points3D_binary
from read_write_model import write_cameras_text, write_images_text, write_points3D_text
from set_database import camTodatabase
neighbor_view_ids = ["000", "001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012",
                                  "013", "014", "015", "016", "017", "018", "019", "020"]

def save_args(args, output_path):
    argsDict = args.__dict__
    output_file = os.path.join(output_path, "args.txt")
    with open(output_file, 'w') as f:
        # f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        # f.writelines('------------------- end -------------------')



def main(args):
    use_gpu = 1 if not args.no_gpu else 0

    ymd = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ymd_dir = f"{ymd}_all" if args.use_all else f"{ymd}_3"
    work_dir = os.path.join("./runs", args.scene_id, ymd_dir)
    os.makedirs(work_dir, exist_ok=True)

    save_args(args, work_dir)

    img_dir = os.path.join(work_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    if args.use_all: # use all image for sfm
        for all_idx in neighbor_view_ids:
            src_img_path = os.path.join(args.dataset_path, args.scene_id, "Images/{}.png".format(all_idx))
            shutil.copy(src_img_path, os.path.join(img_dir, "{}.png".format(all_idx)))
    else:
        for idx in args.train_list:
            src_img_path = os.path.join(args.dataset_path, args.scene_id, "Images/{}.png".format(neighbor_view_ids[idx]))
            shutil.copy(src_img_path, os.path.join(img_dir, "{}.png".format(neighbor_view_ids[idx])))

    # feature extraction
    database_path = os.path.join(work_dir, "database.db")
    colmap_extract_features = [
        'COLMAP.bat', 'feature_extractor',
        '--database_path', database_path,
        '--image_path', img_dir,
        '--ImageReader.single_camera', '1',
        '--SiftExtraction.use_gpu', str(use_gpu),
        '--SiftExtraction.dsp_min_scale', '0.16667',
        '--SiftExtraction.peak_threshold', '0.00668'
    ]
    print(f"Running command: {' '.join(colmap_extract_features)}")
    result_feature_extractor = subprocess.run(colmap_extract_features, capture_output=True, text=True)
    print(f"Command stderr:\n{result_feature_extractor.stderr}")
    print(f"Command stdout:\n{result_feature_extractor.stdout}")
    print(f"Exit status: {result_feature_extractor.returncode}")


    # generate pose
    posed_sparse_model_path = os.path.join(work_dir, "posed_sparse_model")


    # set database camera parameters
    generate_init_txt(args, posed_sparse_model_path)
    camTodatabase(database_path,os.path.join(posed_sparse_model_path, "cameras.txt"))

    # feature matching
    colmap_exhaustive_matcher = [
        'COLMAP.bat', 'exhaustive_matcher',
        '--database_path', database_path,
        '--SiftMatching.max_distance', '0.8',
        '--SiftMatching.use_gpu', str(use_gpu),
    ]

    print(f"Running command: {' '.join(colmap_exhaustive_matcher)}")
    result_exhaustive_matcher = subprocess.run(colmap_exhaustive_matcher, capture_output=True, text=True)
    print(f"Command stderr:\n{result_exhaustive_matcher.stderr}")
    print(f"Command stdout:\n{result_exhaustive_matcher.stdout}")
    print(f"Exit status: {result_exhaustive_matcher.returncode}")

    # triangulated
    triangulated_sparse_model_path = os.path.join(work_dir, "triangulated_sparse_model")
    os.makedirs(triangulated_sparse_model_path)
    colmap_triangulated = [
        'COLMAP.bat', 'point_triangulator',
        '--database_path', database_path,
        '--image_path', img_dir,
        '--input_path', posed_sparse_model_path,
        '--output_path', triangulated_sparse_model_path
    ]

    #  COLMAP
    print(f"Running command: {' '.join(colmap_triangulated)}")
    result_point_triangulator = subprocess.run(colmap_triangulated, capture_output=True, text=True)
    print(f"Command stderr:\n{result_point_triangulator.stderr}")
    print(f"Command stdout:\n{result_point_triangulator.stdout}")
    print(f"Exit status: {result_point_triangulator.returncode}")

    # dense model from sparse model

    dense_model_path = os.path.join(work_dir, "dense")
    os.makedirs(dense_model_path)

    colmap_image_undistorter = [
        'COLMAP.bat', 'image_undistorter',
        '--image_path', img_dir,
        '--input_path', triangulated_sparse_model_path,
        '--output_path', dense_model_path,
    ]
    # COLMAP
    print(f"Running command: {' '.join(colmap_image_undistorter)}")
    result_image_undistorter = subprocess.run(colmap_image_undistorter, capture_output=True, text=True)
    print(f"Command stderr:\n{result_image_undistorter.stderr}")
    print(f"Command stdout:\n{result_image_undistorter.stdout}")
    print(f"Exit status: {result_image_undistorter.returncode}")

    if args.fuse_plt:
        colmap_patch_match = [
            'COLMAP.bat', 'patch_match_stereo',
            '--workspace_path', dense_model_path,
        ]
        print(f"Running command: {' '.join(colmap_patch_match)}")
        result_2 = subprocess.run(colmap_patch_match, text=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        print(f"Command stderr:\n{result_2.stderr}")
        print(f"Command stdout:\n{result_2.stdout}")
        print(f"Exit status: {result_2.returncode}")


        pycolmap.stereo_fusion(os.path.join(dense_model_path,"dense.ply"), dense_model_path)


    if args.convert_to_txt:
        cameras_bin = read_cameras_binary(os.path.join(dense_model_path, "sparse", "cameras.bin"))
        images_bin = read_images_binary(os.path.join(dense_model_path, "sparse", "images.bin"))
        points3D_bin = read_points3D_binary(os.path.join(dense_model_path, "sparse", "points3D.bin"))

        sparse_txt_path = os.path.join(dense_model_path, "sparse_txt")
        os.makedirs(sparse_txt_path)
        write_cameras_text(cameras_bin, os.path.join(sparse_txt_path, "cameras.txt"))
        write_images_text(images_bin, os.path.join(sparse_txt_path, "images.txt"))
        write_points3D_text(points3D_bin, os.path.join(sparse_txt_path, "points3D.txt"))

    # move to other dataset
    if args.mv_to_dataset:
        gs_scene_path = os.path.join(args.dataset_path, args.scene_id)
        tgt_sparse_path = os.path.join(gs_scene_path, "sparse")
        if not os.path.exists(tgt_sparse_path):
            os.makedirs(tgt_sparse_path, exist_ok=True)
        if args.use_all:
            # move to original dataset
            # only move images.bin and cameras.bin
            sparse_path = os.path.join(dense_model_path, "sparse")
            os.makedirs(tgt_sparse_path, exist_ok=True)
            shutil.copytree(sparse_path, os.path.join(tgt_sparse_path, "0"))
            print("Moved all views poses files!")
        elif args.fuse_plt:
            # only move dense.ply and replace points3D.bin with 3-views
            dense_ply_src_path = os.path.join(dense_model_path, "dense.ply")
            dense_ply_tgt_path = os.path.join(tgt_sparse_path, "0", "dense.ply")
            shutil.copyfile(dense_ply_src_path, dense_ply_tgt_path)
            # replace points3D.bin
            points3D_src_path = os.path.join(dense_model_path, "sparse", "points3D.bin")
            points3D_tgt_path = os.path.join(tgt_sparse_path, "0", "points3D.bin")
            if os.path.exists(points3D_tgt_path):
                os.remove(points3D_tgt_path)
                # delete ply at the same time
                ply_path = os.path.join(tgt_sparse_path, "0", "points3D.ply")
                if os.path.exists(ply_path):
                    os.remove(ply_path)
            shutil.copyfile(points3D_src_path, points3D_tgt_path)
            print("Replaced points3D.bin with 3-views.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", help="path to dataset", type=str, default="../data/LEVIR_NVS")
    parser.add_argument("-s", "--scene_id", help="scene id",type=str, default="scene_000")
    parser.add_argument("-t", "--train_list", type=int, default=[1, 8, 15], nargs='*', help='training_idx')
    parser.add_argument("--mv_to_dataset", action="store_true") # whether move sfm files to dataset
    parser.add_argument("--fuse_plt", action="store_true") # whether to fuse point cloud using patch match stereo
    parser.add_argument("--use_all", action="store_true", help='use all images')
    parser.add_argument("--convert_to_txt", action="store_true", help='convert .bin files to .txt')
    parser.add_argument("--no_gpu", action="store_true", help='use gpu')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

