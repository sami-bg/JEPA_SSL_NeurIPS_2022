from dataclasses import dataclass
from typing import Optional, Any, Literal
import copy
import torch
from torch.nn import functional as F
import wandb
import numpy as np
from matplotlib import pyplot as plt
import utils
from configs import ConfigBase

import models

@dataclass
class VJEPAConfig(ConfigBase):
    arch: str = "vit"
    # model_type: str = 'VJEPA'
    #### EMA
    ema: tuple[float, float] = (0.998, 1.0)
    ipe_scale: float = 1.25
    ipe: int = 10_000
    #### Data
    masking_ratio: float = 0.25
    img_size: int = 28
    patch_size: int = 4
    num_frames: int = 18
    tubelet_size: int = 2
    channels: int = 1

    #### Optim
    epochs: int = 100
    base_lr: float = 0.2
    
    #### Downstream Predictor
    predictor: str = "attentive_classifier"
    attentive_num_heads: int = 2
    attentive_depth: int = 2
    action_dim: int = 2

    #### JEPA Predictor
    predictor_embed_dim: int = 32
    predictor_depth: int = 4
    predictor_num_heads: int = 4
    predictor_mlp_ratio: float = 4.0
    predictor_qkv_bias: bool = True
    predictor_qk_scale: Optional[float] = None
    predictor_drop_rate: float = 0.0
    predictor_attn_drop_rate: float = 0.0
    predictor_drop_path_rate: float = 0.0

    #### Encoder
    encoder_embed_dim: int = 64
    encoder_depth: int = 6
    encoder_num_heads: int = 4
    encoder_mlp_ratio: float = 4.0
    encoder_qkv_bias: bool = True
    encoder_qk_scale: Optional[float] = None
    encoder_drop_rate: float = 0.0
    encoder_attn_drop_rate: float = 0.0
    encoder_drop_path_rate: float = 0.0
    #### Misc
    rnn_burnin: int = 1
    temporal_inconsistency_enabled: bool = True
    temporal_inconsistency_coeff: float = 1.0
    # NOTE One of "full" or "pairwise", but we can't annotate with Literal because of omegaconf
    temporal_inconsistency_type: str = "full"

@dataclass
class LossInfo:
    total_loss: torch.Tensor
    diagnostics_info: Any

    
class VJEPA(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.masking_ratio = args.masking_ratio
        self.num_patches_spatial = args.img_size // args.patch_size
        self.num_frames = args.num_frames
        self.num_keep_spatial = int(self.num_patches_spatial * (1 - args.masking_ratio))
        self.backbone, self.embedding = models.build_backbone(
            args.arch,
            args.encoder_embed_dim,
            backbone_mlp=None,
            backbone_width_factor=1,
            channels=args.channels,
            **{
                "img_size": args.img_size,
                "patch_size": args.patch_size,
                "num_frames": args.num_frames,
                "tubelet_size": args.tubelet_size,
                "encoder_embed_dim": args.encoder_embed_dim,
                "encoder_depth": args.encoder_depth,
                "encoder_num_heads": args.encoder_num_heads,
                "encoder_mlp_ratio": args.encoder_mlp_ratio,
                "encoder_qkv_bias": args.encoder_qkv_bias,
                "encoder_qk_scale": args.encoder_qk_scale,
            }
        )
        self.predictor = models.build_predictor(
            args.predictor, self.embedding, self.args.action_dim,
            rnn_layers=0,
            attentive_num_heads=args.attentive_num_heads,
            attentive_depth=args.attentive_depth
        )
        print(f"backbone is {args.arch}")
        self.backbone: models.VisionTransformer
        ema, ipe, ipe_scale, num_epochs = args.ema, args.ipe, args.ipe_scale, args.epochs
        self.ema_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))
        self.ema_backbone = copy.deepcopy(self.backbone)
        self.vit_predictor = models.VisionTransformerPredictor(
            tubelet_size=args.tubelet_size,
            img_size=args.img_size,
            patch_size=args.patch_size,
            num_frames=args.num_frames,
            embed_dim=args.encoder_embed_dim,
            predictor_embed_dim=args.predictor_embed_dim,
        )

    def l2_loss(self, predicted_representations, target_representations):
        loss = 0.
        # Compute loss and accumulate for each mask-enc/mask-pred pair
        for zi, hi in zip(predicted_representations, target_representations):
            loss += torch.mean(torch.abs(zi - hi)**2) / 2
        # I think there is just 1 mask-enc/mask-pred pair
        # loss /= len(masks_pred)
        return loss

    def sample_mask(self):
        mask = np.hstack([
            np.zeros(self.num_patches_spatial - self.num_keep_spatial),
            np.ones(self.num_keep_spatial),
        ])
        np.random.shuffle(mask)
        mask = torch.tensor(np.tile(mask, (self.num_frames, 1)))
        mask = mask.flatten()
        mask_p = torch.argwhere(mask == 0).squeeze()
        mask_e = torch.nonzero(mask).squeeze()
        return mask_e, mask_p
    
    def tube_masking(self, batch_size: int):
        collated_masks_pred, collated_masks_enc = [], []
        for _ in range(batch_size):
            mask_e, mask_p = self.sample_mask()
            collated_masks_enc.append(mask_e.to(self.device))
            collated_masks_pred.append(mask_p.to(self.device))

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        return collated_masks_enc, collated_masks_pred

    def temporal_inconsistency_loss(self, patches):
        assert self.args.temporal_inconsistency_enabled
        assert self.args.temporal_inconsistency_type in ["full", "pairwise"]
        loss = 0.
        tubelets_per_frame = self.num_frames // self.backbone.tubelet_size
        patches_per_tubelet = patches.shape[1] // tubelets_per_frame
        # NOTE Need to do some arithmetic here to take one patch across all frames
        # NOTE If tubelet size is 2, num_frames is 18, and I have 36 patches,it means
        # there are 4 context patches in each image. So if I want to do dissimilarity
        # for the same image position, then I need to stride by 4 each time across the patch
        # dimension.
        for spatial_position in range(patches_per_tubelet):
            patch_across_time = patches[:, spatial_position::patches_per_tubelet, ...]
            if self.args.temporal_inconsistency_type == "full":
                # NOTE Each spatial-patch should be maximally different from every other patch across time.
                # Each patch should be unique, but this could collapse everything with fixed noise because
                # we could just be masking the movingdot!
                for temporal_position in range(patch_across_time.shape[1]):
                    for other_temporal_position in range(temporal_position+1, patch_across_time.shape[1]):
                        p_i, p_j = (patch_across_time[:, temporal_position, ...],
                                    patch_across_time[:, other_temporal_position, ...])
                        loss += torch.cosine_similarity(p_i, p_j).mean()
            if self.args.temporal_inconsistency_type == "pairwise":
                # NOTE Each spatial-patch should be different from the next frame's corresponding patch.
                # TODO Should we add a frame-skip?
                for temporal_position in range(patch_across_time.shape[1]-1):
                    p_i, p_i_next = (patch_across_time[:, temporal_position, ...],
                                     patch_across_time[:, temporal_position+1, ...])
                    loss += torch.cosine_similarity(p_i, p_i_next).mean()
        return self.args.temporal_inconsistency_coeff * loss

    def forward(self, states, actions, step=None):
        """states [T, batch_size, 1, 28, 28]
        actions [T-1, batch_size]
        """

        T,B,C,H,W = states.shape
        ##### Tube masking
        # Might need to change some of this since original code assumes B,T,...
        masks_enc, masks_pred = self.tube_masking(B)
        ##### Forward target
        with torch.no_grad():
            target_patches  = self.ema_backbone(states, masks=masks_pred)
        ##### Forward context
        context_patches = self.backbone(states, masks=masks_enc)
        # TODO Somehow condition the predictor on the actions
        predicted_target_patches = self.vit_predictor(context_patches, target_patches, masks_enc, masks_pred, actions)
        ##### Compute loss
        loss = self.l2_loss(predicted_target_patches, target_patches)
        if self.args.temporal_inconsistency_enabled:
            loss += self.temporal_inconsistency_loss(predicted_target_patches)
            loss += self.temporal_inconsistency_loss(context_patches)

        #### EMA for target
        m = next(self.ema_scheduler)
        with torch.no_grad():
            for param_q, param_k in zip(self.backbone.parameters(), self.ema_backbone.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        return LossInfo(total_loss=loss, diagnostics_info=None)

        
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