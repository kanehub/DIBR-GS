import functools
from dataclasses import dataclass
from typing import Literal

import torch
import torchvision
from torch import nn
from torchvision.models import ResNet


@dataclass
class BackboneResnetCfg:
    name: Literal["resnet"]
    model: Literal[
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "dino_resnet50"
    ]
    num_layers: int
    use_first_pool: bool
    d_out: int


class BackboneResnet(nn.Module):
    Model: ResNet
    def __init__(self, cfg: BackboneResnetCfg, d_in: int) -> None:
        super().__init__()

        assert d_in == 3

        self.cfg = cfg

        norm_layer = functools.partial(
            nn.InstanceNorm2d,
            affine=False,
            track_running_stats=False,
        )

        if cfg.model == "dino_resnet50":
            self.model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
        else:
            self.model = getattr(torchvision.models, cfg.model)(norm_layer=norm_layer)


    def forward(
        self,
        x,
    ):

        # Run the images through the resnet.
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        feature_dict = {}
        feature_dict['layer0'] = x
        for index in range(1, self.cfg.num_layers):
            key = f"layer{index}"
            if index == 0 and self.cfg.use_first_pool:
                x = self.model.maxpool(x)
            x = getattr(self.model, key)(x)
            feature_dict[key] = x

        return feature_dict

    @property
    def d_out(self) -> int:
        return self.cfg.d_out
