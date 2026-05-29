from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from modeling.backbone_resnet import BackboneResnet, BackboneResnetCfg


@dataclass
class BackboneDinoCfg:
    name: Literal["dino"]
    model: Literal["dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8"]
    d_out: int


class BackboneDino(nn.Module):
    def __init__(self, cfg:BackboneDinoCfg, d_in: int) -> None:
        super().__init__()
        assert d_in == 3
        self.cfg = cfg
        self.dino = torch.hub.load("facebookresearch/dino:main", cfg.model)
        self.resnet_backbone = BackboneResnet(
            BackboneResnetCfg("resnet", "dino_resnet50", 4, False, cfg.d_out),
            d_in,
        )
        self.global_token_mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, cfg.d_out),
        )
        self.local_token_mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, cfg.d_out),
        )

    def forward(
        self,
        x,
    ):
        # Compute features from the DINO-pretrained resnet50.
        resnet_features_dict = self.resnet_backbone(x)

        # Compute features from the DINO-pretrained ViT.
        b, c, h, w = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0

        tokens = self.dino.get_intermediate_layers(x)[0]

        return resnet_features_dict, tokens

    @property
    def patch_size(self) -> int:
        return int("".join(filter(str.isdigit, self.cfg.model)))

    @property
    def d_out(self) -> int:
        return self.cfg.d_out
