import torch
import torch.nn.functional as F
import numpy as np
import math


def get_c2w_poses(camera):
    opencv2blender = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    C2W = C2W @ opencv2blender

    # return c2w opengl
    return np.float32(C2W)

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


class Projector():
    def __init__(
            self,
            train_cameras,
            device,
            padding_mode="zeros"
    ):
        camera_list = train_cameras[1.0] # original resolution
        images = [cam.original_image for cam in camera_list] # List[ Tensor(3,H,W)]

        poses = [get_c2w_poses(cam) for cam in camera_list]
        c2w = np.stack(poses, axis=0) # [N, 4, 4]

        img_size, _ = images[0].shape[1:3] # square image
        f = fov2focal(camera_list[0].FoVx, img_size)  # fovx == fovy

        train_images = torch.stack(images, dim=0) # [N,3,H,W]

        self.train_images = train_images

        # self.reference = reference
        self.scale = (img_size / 2) / f
        self.n = c2w.shape[0]
        self.R_t = torch.tensor(c2w[:, :3, :3], device=device).permute(0, 2, 1) #[3,3,3] w2c
        self.camera_pos = torch.tensor(c2w[:, :3, -1], device=device)
        # self.c2w = c2w
        self.img_size = img_size
        self.f = f
        self.ref_feature = None
        self.ref_rgb = None
        self.n_refs = 3

        self.neareat_idxs = None
        self.padding_mode = padding_mode

    def feature_matching(self, pos):
        # pos (world)  opengl coord
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(0) # [1,N,3]
        n_rays, n_samples, _ = pos.shape
        pos = pos.unsqueeze(dim=0).expand([self.n, n_rays, n_samples, 3]) # [3,nr,ns,3]
        camera_pos = self.camera_pos[:, None, None, :] # [3,1,1,3]
        camera_pos = camera_pos.expand_as(pos) # [3,nr,ns,3]
        ref_pos = torch.einsum("kij,kbsj->kbsi", self.R_t, pos-camera_pos) # [3, nr,ns, 3]
        uv_pos = ref_pos[..., :-1] / -ref_pos[..., -1:] / self.scale
        uv_pos[..., 1] *= -1.0

        if self.n_refs <= 3:
            nearest_ref_feature = self.ref_feature
        else:
            nearest_ref_feature = self.ref_feature[self.neareat_idxs]

        n_feat_vec = F.grid_sample(nearest_ref_feature, uv_pos, align_corners=True, padding_mode=self.padding_mode)
        avg_feat_vec = torch.mean(n_feat_vec, dim=0)

        ref_feat_vec = avg_feat_vec.permute(1, 2, 0).squeeze()
        return ref_feat_vec

    def update_ref_feature(self, ref_feature):
        self.ref_feature = ref_feature
        self.n_refs = self.ref_feature.shape[0]

    def update_ref_rgb(self, ref_rgb):
        self.ref_rgb = ref_rgb

    def rgb_matching(self, pos):
        # pos (world)  opengl coord
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(0) # [1,N,3]
        n_rays, n_samples, _ = pos.shape
        pos = pos.unsqueeze(dim=0).expand([self.n, n_rays, n_samples, 3]) # [3,nr,ns,3]
        camera_pos = self.camera_pos[:, None, None, :] # [3,1,1,3]
        camera_pos = camera_pos.expand_as(pos) # [3,nr,ns,3]
        ref_pos = torch.einsum("kij,kbsj->kbsi", self.R_t, pos-camera_pos) # [3, nr,ns, 3]
        uv_pos = ref_pos[..., :-1] / -ref_pos[..., -1:] / self.scale
        uv_pos[..., 1] *= -1.0

        if self.n_refs <= 3:
            nearest_ref_feature = self.ref_rgb
        else:
            nearest_ref_feature = self.ref_rgb[self.neareat_idxs]

        n_feat_vec = F.grid_sample(nearest_ref_feature, uv_pos, align_corners=True, padding_mode=self.padding_mode)
        avg_feat_vec = torch.mean(n_feat_vec, dim=0)

        ref_feat_vec = avg_feat_vec.permute(1, 2, 0).squeeze()
        return ref_feat_vec

    def update_nearest_idx(self, current_idx):

        neareat_idxs = [(current_idx-1)%self.n_refs, current_idx, (current_idx+1)%self.n_refs]
        self.neareat_idxs = neareat_idxs