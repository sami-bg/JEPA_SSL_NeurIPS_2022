from dataclasses import dataclass
from typing import Optional, Any, Literal
import copy
import torch
from torch.nn import functional as F
import wandb
import numpy as np
from matplotlib import pyplot as plt
import utils
from utils import apply_masks
from configs import ConfigBase

import models

@dataclass
class HJEPAConfig(ConfigBase):
    arch: str = "vit"
    # model_type: str = 'HJEPA'
    #### EMA
    ema: tuple[float, float] = (0.998, 1.0)
    ipe_scale: float = 1.25
    ipe: int = 10_000
    #### Data
    img_size: int = 28
    num_frames: int = 16
    channels: int = 1
    patch_size: int = 4
    masking_ratio: float = 0.7
    #### Hierarchy (map 1-to-1 per hierarchy)
    tubelet_sizes: tuple[int] = (2, 4)
    hier_attentive_num_heads: tuple[int] = (4,) # NOTE embed_dim % num_heads == 0

    #### Optim
    epochs: int = 100
    base_lr: float = 0.2
    
    #### Downstream Predictor
    predictor: str = "attentive_classifier"
    downstream_attentive_num_heads: int = 2
    downstream_attentive_depth: int = 2
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
    predictor_use_mask_tokens: bool = True
    predictor_zero_init_mask_tokens: bool = True
    predictor_num_mask_tokens: int = 1

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


@dataclass
class LossInfo:
    total_loss: torch.Tensor
    diagnostics_info: Any


class HJEPA(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # -- video data
        self.num_frames = args.num_frames
        self.img_size = args.img_size
        # -- hierarchy
        # Validate that we have the same number of hierarchies for each
        assert tuple(sorted(args.tubelet_sizes)) == tuple(args.tubelet_sizes)
        assert all(self.num_frames % t == 0 for t in args.tubelet_sizes), "each hierarchy must be divisible by the number of frames"
        self.num_hierarchies = len(args.tubelet_sizes)
        self.masking_ratio = args.masking_ratio
        self.patch_size: int = args.patch_size
        self.num_patches_spatial: int = (args.img_size // self.patch_size)**2
        self.num_keep_spatial: int = int(self.num_patches_spatial * (1 - self.masking_ratio))
        # -- hierarchy
        self.tubelet_sizes: tuple[int, ...] = args.tubelet_sizes
        self.grid_depths: tuple[int, ...] = tuple(self.num_frames // t for t in self.tubelet_sizes)
        # -- build backbones for each level of the hierarchy
        self.backbones, self.embeddings = zip(*(
            models.build_backbone(
                args.arch,
                args.encoder_embed_dim,
                backbone_mlp=None,
                backbone_width_factor=1,
                **{
                    "img_size": ims,
                    "patch_size": ps,
                    "num_frames": nf,
                    "tubelet_size": ts,
                    "encoder_embed_dim": eed,
                    "encoder_depth": ed,
                    "encoder_mlp_ratio": emlpr,
                    "encoder_qkv_bias": eqkvb,
                    "encoder_qk_scale": eqks,
                    "encoder_num_heads": nh,
                }
            )
            for ps, ts, nf, ims, eed, ed, emlpr, eqkvb, eqks, nh in zip(
                (self.patch_size,) * self.num_hierarchies,
                self.tubelet_sizes,
                (self.num_frames,) * self.num_hierarchies,
                (self.img_size,) * self.num_hierarchies,
                (args.encoder_embed_dim,) * self.num_hierarchies,
                (args.encoder_depth,) * self.num_hierarchies,
                (args.encoder_mlp_ratio,) * self.num_hierarchies,
                (args.encoder_qkv_bias,) * self.num_hierarchies,
                (args.encoder_qk_scale,) * self.num_hierarchies,
                (args.encoder_num_heads,) * self.num_hierarchies,
            )
        ))
        for bb in self.backbones: bb = bb.cuda()

        # NOTE Downstream predictor, not vit predictor for jepa
        self.predictor = models.build_predictor(
            args.predictor, self.embeddings[0],  # NOTE We have to specify which hierarchy we are using, this assumes 0 (tubesize=2)
            self.args.action_dim,
            rnn_layers=0, 
            attentive_num_heads=args.downstream_attentive_num_heads,
            attentive_depth=args.downstream_attentive_depth
        )
        self.predictor = self.predictor.cuda()

        self.ema_schedulers = [
            (args.ema[0] + i*(args.ema[1]-args.ema[0])/(args.ipe*args.epochs*args.ipe_scale)
                          for i in range(int(args.ipe*args.epochs*args.ipe_scale)+1))
            for _ in range(self.num_hierarchies)
        ]
        self.ema_backbones: list[torch.nn.Module] = [copy.deepcopy(self.backbones[i]) for i in range(self.num_hierarchies)]
        for bb in self.ema_backbones: 
            for p in bb.parameters(): p.requires_grad = False

        self.vit_v_predictors = [
            models.VisionTransformerPredictor(tubelet_size=ts, img_size=ims, patch_size=ps, num_frames=nf,
                                             use_mask_tokens=args.predictor_use_mask_tokens,
                                             num_mask_tokens=args.predictor_num_mask_tokens,
                                             zero_init_mask_tokens=args.predictor_zero_init_mask_tokens)
            for ts, ps, ims, nf in zip(
                self.tubelet_sizes, (self.patch_size,) * self.num_hierarchies,
                (self.img_size,) * self.num_hierarchies, (self.num_frames,) * self.num_hierarchies
            )
        ]
        for v in self.vit_v_predictors: v = v.cuda()

        # NOTE Ratio of tubelet size of i+1 to i
        self.h_tubelet_ratios = tuple(t2//t1 for t1, t2 in zip(self.tubelet_sizes, self.tubelet_sizes[1:]))
        assert len(self.h_tubelet_ratios) == len(self.args.hier_attentive_num_heads) == len(self.embeddings) - 1

        self.attentive_h_predictors = [
            models.AttentivePooler(
                num_queries=self.num_patches_spatial * self.grid_depths[i+1],  # NOTE Next num patches
                out_dim=self.embeddings[i+1],
                depth=1,
                num_heads=self.args.hier_attentive_num_heads[i],
                # NOTE Check why this is not necessary? Or if it is?
                embed_dim=self.embeddings[i] # * self.h_tubelet_ratios[i]  # NOTE Predict H-4 from 2 H-2s
            )
            for i in range(self.num_hierarchies-1)
        ]
        for a in self.attentive_h_predictors: a = a.cuda()
        assert len(set(map(len, [
            self.vit_v_predictors,
            self.ema_schedulers,
            self.ema_backbones,
            self.backbones,
            self.embeddings]))) == 1

    def l2_loss(self, predicted_representations, target_representations):
        loss = 0.
        for zi, hi in zip(predicted_representations, target_representations):
            loss += torch.mean(torch.abs(zi - hi)**2) / 2
        return loss

    def sample_spatial_mask(self):
        mask = np.hstack([
            np.zeros(self.num_patches_spatial - self.num_keep_spatial),
            np.ones(self.num_keep_spatial),
        ])
        np.random.shuffle(mask)
        return mask
    
    def temporal_maskify(self, spatial_mask: np.ndarray, hierarchy_idx: int):
        mask = torch.tensor(np.tile(spatial_mask, (self.grid_depths[hierarchy_idx], 1)))
        mask = mask.flatten()
        mask_e = torch.nonzero(mask).squeeze()
        mask_p = torch.argwhere(mask == 0).squeeze()
        return mask_e, mask_p
    
    def tube_masking(self, batch_size: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        collated_masks_pred = [list() for _ in range(self.num_hierarchies)]
        collated_masks_enc = [list() for _ in range(self.num_hierarchies)]
        # NOTE Each hierarchy shares the same spatial mask
        spatial_mask = self.sample_spatial_mask()
        for _ in range(batch_size):
            masks_e, masks_p = zip(*[self.temporal_maskify(spatial_mask, i) for i in range(self.num_hierarchies)])
            for i in range(self.num_hierarchies):
                collated_masks_enc[i].append(masks_e[i].to(self.device))
                collated_masks_pred[i].append(masks_p[i].to(self.device))

        collated_masks_enc = [torch.utils.data.default_collate(collated_masks_enc[i]) for i in range(self.num_hierarchies)]
        collated_masks_pred = [torch.utils.data.default_collate(collated_masks_pred[i]) for i in range(self.num_hierarchies)]
        # list of tensors, one per hierarchy
        return collated_masks_enc, collated_masks_pred
    
    def forward(self, states, actions, step=None):
        """states [T, batch_size, 1, 28, 28]
        actions [T-1, batch_size]
        """
        T,B,C,H,W = states.shape
        
        ##### Tube masking
        # These are of shape [batch_size, num_hierarchies, num_patches, 3]
        masks_enc, masks_pred = self.tube_masking(B)
        loss = 0.
        # NOTE: Logging
        hierarchy_loss_stats = {}

        hierarchy_patches = []
        # Forward hierarchies
        for i in range(self.num_hierarchies):
            # Each hierarchy has its own masks, so we access the correct ones
            masks_pred_i, masks_enc_i = masks_pred[i], masks_enc[i]
            ##### Forward target
            with torch.no_grad():
                target_patches  = self.ema_backbones[i](states, masks=None)
                target_patches = F.layer_norm(target_patches, (target_patches.size(-1),))  # normalize over feature-dim  [B, N, D]
                target_patches = apply_masks(target_patches, masks_pred_i)
            ##### Forward context
            context_patches = self.backbones[i](states, masks=masks_enc_i)
            predicted_target_patches = self.vit_v_predictors[i](context_patches, target_patches, masks_enc_i, masks_pred_i, actions)
            loss += (h_loss := self.l2_loss(predicted_target_patches, target_patches))
            hierarchy_loss_stats[f"h{i}_reconstruction_loss"] = h_loss
            hierarchy_patches.append(predicted_target_patches)
        
        # Next-hierarchy distillation
        for i in range(self.num_hierarchies-1):
            # TODO 4/21/2025 This should predict overall patches, not just target patches? Otherwise how would we align
            # the num patches? Answer: with the masks! 
            predicted_next_hierarchy = self.attentive_h_predictors[i](hierarchy_patches[i])
            predicted_next_hierarchy = apply_masks(predicted_next_hierarchy, masks_pred[i+1])
            # NOTE predicted_next_hierarchy is of shape [B, 49, d]
            # NOTE there is a bug because the masks are of different sizes for each hierarchy
            # this is unavoidable because each hierarchy has a different number of patches because of the tubelet size
            # need to draw this out to see how we should fix this.
            loss += (h_loss := self.l2_loss(predicted_next_hierarchy, hierarchy_patches[i+1]))
            hierarchy_loss_stats[f"h{i}to{i+1}_pred_loss"] = h_loss
        
        # EMA for each hierarchy
        for i in range(self.num_hierarchies):
            m = next(self.ema_schedulers[i])
            with torch.no_grad():
                for param_q, param_k in zip(self.backbones[i].parameters(), self.ema_backbones[i].parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        return LossInfo(total_loss=loss, diagnostics_info=hierarchy_loss_stats)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = torch.randn(18, 1, 1, 28, 28).to(device)
    actions = torch.randn(17, 1).to(device)
    hjepa = HJEPA(HJEPAConfig()).cuda()
    loss = hjepa(states, actions)
    print(loss)
    