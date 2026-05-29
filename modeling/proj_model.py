import os
from torch import nn
import torch
from torch.nn import functional as F

from einops import repeat

from utils_loc.system_utils import searchForMaxIteration
from utils_loc.general_utils import get_expon_lr_func


class ProjectMLP(nn.Module):
    def __init__(self, d_out: int = 512, resnet_num_layers=4, patch_size = 8):
        super().__init__()

        resnet50_channels = [64, 256, 512, 1024]
        self.patch_size = patch_size

        self.projections = nn.ModuleDict({})

        for index in range(resnet_num_layers):
            key = f"layer{index}"
            d_layer_out = resnet50_channels[index]
            self.projections[key] = nn.Conv2d(d_layer_out, d_out, 1)

        # vit
        self.global_token_mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, d_out),
        )
        self.local_token_mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, d_out),
        )
    def forward(self, resnet_features: dict[str, torch.Tensor], vit_tokens: torch.Tensor):

        projected_feats = self.resnet_proj(resnet_features)

        b, c, h, w = projected_feats.shape

        global_token = self.global_token_mlp(vit_tokens[:, 0])
        local_tokens = self.local_token_mlp(vit_tokens[:, 1:])

        # Repeat the global token to match the image shape.
        global_token = repeat(global_token, "b c -> b c h w", b=b, h=h, w=w)

        # Repeat the local tokens to match the image shape.
        local_tokens = repeat(
            local_tokens,
            "b (h w) c -> b c (h hps) (w wps)",
            b=b,
            h=2 * h // self.patch_size,
            hps=self.patch_size // 2,
            w=2 * w // self.patch_size,
            wps=self.patch_size // 2,
        )

        return projected_feats + local_tokens + global_token

    def resnet_proj(self, features: dict):
        projected_feats = []
        _, _, h, w = features["layer0"].shape
        for index in range(len(self.projections)):
            key = f"layer{index}"
            feat = features[key]
            proj_feat = self.projections[key](feat)
            _, _, hc, wc = proj_feat.shape
            if hc != h or wc != w:
                proj_feat = F.interpolate(proj_feat, (h, w), mode="bilinear", align_corners=True)
            projected_feats.append(proj_feat)

        projected_feats = torch.stack(projected_feats).sum(dim=0)
        return projected_feats

class  ProjectModel():
    def __init__(self, args, opt=None):

        self.args = args
        self.opt = opt

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ProjectMLP(d_out=args.ref_feat_dim).to(device)

        if opt is not None:
            self.optimizer = self.get_optimizer()

            self.scheduler_func = get_expon_lr_func(lr_init=opt.mlp_proj_lr_init,
                                                    lr_final=opt.mlp_proj_lr_final,
                                                    lr_delay_steps=opt.mlp_proj_lr_delay_steps,
                                                    lr_delay_mult=opt.mlp_proj_lr_delay_mult,
                                                    max_steps=opt.mlp_proj_lr_max_steps)

    def get_optimizer(self):
        lr = self.opt.mlp_proj_lr_init
        params_list = [dict(params=self.model.parameters(), lr=lr)]
        optim = torch.optim.AdamW(
            params_list,
            weight_decay=1e-5,
            eps=1e-8,
        )
        return optim

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            lr = self.scheduler_func(iteration)
            param_group['lr'] = lr

    def save(self, iteration):
        point_cloud_path = os.path.join(self.args.model_path, "point_cloud/iteration_{}".format(iteration))
        torch.save(self.model.state_dict(), os.path.join(point_cloud_path, "proj_mlp.pth"))
        print(f"Save project_model to {point_cloud_path}")

    def load(self, iteration):

        if iteration == -1:
            iteration = searchForMaxIteration(os.path.join(self.args.model_path, "point_cloud"))


        point_cloud_path = os.path.join(self.args.model_path, "point_cloud/iteration_{}".format(iteration))
        proj_state_dict = torch.load(os.path.join(point_cloud_path, "proj_mlp.pth"), map_location="cuda")
        self.model.load_state_dict(proj_state_dict)
        print(f"Load project_model from {point_cloud_path}")

    def switch_to_eval(self):
        self.model.eval()

    def switch_to_train(self):
        self.model.train()
