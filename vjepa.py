from dataclasses import dataclass
from typing import Optional, Any

import torch
from torch.nn import functional as F
import wandb
import numpy as np
from matplotlib import pyplot as plt

from configs import ConfigBase

import models

@dataclass
class VJEPAConfig(ConfigBase):
    patch_size: int = 4
    img_size: int = 28
    num_frames: int = 18
    channels: int = 1

    epochs: int = 100
    base_lr: float = 0.2
    
    predictor_embed_dim: int = 32
    predictor_depth: int = 6
    predictor_num_heads: int = 4
    predictor_mlp_ratio: float = 4.0
    predictor_qkv_bias: bool = True
    predictor_qk_scale: Optional[float] = None
    predictor_drop_rate: float = 0.0
    predictor_attn_drop_rate: float = 0.0
    predictor_drop_path_rate: float = 0.0
    
    encoder_embed_dim: int = 64
    encoder_depth: int = 12
    encoder_num_heads: int = 4
    encoder_mlp_ratio: float = 4.0
    encoder_qkv_bias: bool = True
    encoder_qk_scale: Optional[float] = None
    encoder_drop_rate: float = 0.0
    encoder_attn_drop_rate: float = 0.0
    encoder_drop_path_rate: float = 0.0
    

@dataclass
class LossInfo:
    total_loss: torch.Tensor
    diagnostics_info: Any

    
class VJEPA(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.masking_ratio = args.masking_ratio
        
        self.backbone, self.embedding = models.build_backbone(
            args.arch,
            args.embedding_size,
            args.backbone_mlp,
            args.backbone_width_factor,
            channels=args.channels,
            **{
                "encoder_embed_dim": args.encoder_embed_dim,
                "encoder_depth": args.encoder_depth,
                "encoder_num_heads": args.encoder_num_heads,
                "encoder_mlp_ratio": args.encoder_mlp_ratio,
                "encoder_qkv_bias": args.encoder_qkv_bias,
                "encoder_qk_scale": args.encoder_qk_scale,
            }
        )

        print(f"backbone is {args.arch}")

        self.vit_predictor = models.VisionTransformerPredictor(
            img_size=args.img_size,
            patch_size=args.patch_size,
            num_frames=args.num_frames,
            embed_dim=args.embed_dim,
            predictor_embed_dim=args.predictor_embed_dim,
        )

    def l2_loss(self, predicted_representations, target_representations):
        # TODO patch-wise loss
        return sum(
            F.l1_loss(_pred, _target, reduction="mean")
            for _pred, _target
            in zip(predicted_representations, target_representations)
        ) / len(predicted_representations)

    def tube_masking(self, x):
        # TODO
        return x

    def forward(self, states, actions, step=None):
        """states [T, batch_size, 1, 28, 28]
        actions [T-1, batch_size]
        """
        T,B,C,H,W = states.shape
        # TODO:
        # mask/patchify
        # encode context and target
        # predict target
        # compute loss
        return LossInfo(total_loss=0.0, diagnostics_info=None)
    

        
if __name__ == "__main__":
    args = object()
    args.arch = "vit"
    args.embedding_size = 64
    args.backbone_mlp = None
    args.backbone_width_factor = 1
    args.channels = 1
    args.encoder_embed_dim = 64
    args.encoder_depth = 12
    args.encoder_num_heads = 4
    args.encoder_mlp_ratio = 4.0
    args.encoder_qkv_bias = True
    args.encoder_qk_scale = None
    
    backbone, embedding = models.build_backbone(
        args.arch,
        args.embedding_size,
        args.backbone_mlp,
        args.backbone_width_factor,
        channels=args.channels,
        **{
            "encoder_embed_dim": args.encoder_embed_dim,
            "encoder_depth": args.encoder_depth,
            "encoder_num_heads": args.encoder_num_heads,
            "encoder_mlp_ratio": args.encoder_mlp_ratio,
            "encoder_qkv_bias": args.encoder_qkv_bias,
            "encoder_qk_scale": args.encoder_qk_scale,
        }
    )
    
    print(backbone)