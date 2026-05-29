import torch.nn as nn

from modeling.backbone_dino import BackboneDino, BackboneDinoCfg

class VitEncoder(nn.Module):
    def __init__(
            self,
            output_dim: int = 64
    ) -> None:
        super(VitEncoder, self).__init__()

        self.dino_encoder = BackboneDino(
            BackboneDinoCfg("dino", "dino_vitb8", 512),3)

        self.output_dim = output_dim

    def forward(self, x):
        latents = self.dino_encoder(x)
        return latents