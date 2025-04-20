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
class HJEPAConfig(ConfigBase):
    arch: str = "vit"
    # model_type: str = 'HJEPA'
    #### EMA
    ema: tuple[float, float] = (0.998, 1.0)
    ipe_scale: float = 1.25
    ipe: int = 10_000
    #### Data
    img_size: int = 28
    num_frames: int = 18
    channels: int = 1
    #### Hierarchy (map 1-to-1 per hierarchy)
    patch_sizes: tuple[int] = (4, 4)
    tubelet_sizes: tuple[int] = (2, 4)
    masking_ratios: tuple[float] = (0.7, 0.7)

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

@dataclass
class LossInfo:
    total_loss: torch.Tensor
    diagnostics_info: Any


class HJEPA(torch.nn.Module):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # -- video data
        self.num_frames = args.num_frames
        self.img_size = args.img_size
        # -- hierarchy
        # Validate that we have the same number of hierarchies for each
        assert len(args.patch_sizes) == len(args.tubelet_sizes) == len(args.masking_ratios)
        assert sorted(args.tubelet_sizes) == args.tubelet_sizes
        self.num_hierarchies = len(args.patch_sizes)
        self.masking_ratios = sorted(args.masking_ratios)
        self.patch_sizes: tuple[int, ...] = args.patch_sizes
        self.tubelet_sizes: tuple[int, ...] = args.tubelet_sizes
        self.num_patches_spatial: tuple[int, ...] = tuple(args.img_size // p for p in self.patch_sizes)
        self.nums_keep_spatial: tuple[int, ...] = tuple(
            int(self.num_patches_spatial[i] * (1 - self.masking_ratios[i]))
            for i in range(self.num_hierarchies)
        )
        # -- build backbones for each level of the hierarchy
        self.backbones, self.embeddings = zip(*(
            models.build_backbone(
                args.arch,
                args.encoder_embed_dim,
                backbone_mlp=None,
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
                }
            )
            for ps, ts, nf, ims, eed, ed, emlpr, eqkvb, eqks in zip(
                self.patch_sizes,
                self.tubelet_sizes,
                (self.num_frames,) * self.num_hierarchies,
                (self.img_size,) * self.num_hierarchies,
                (args.encoder_embed_dim,) * self.num_hierarchies,
                (args.encoder_depth,) * self.num_hierarchies,
                (args.encoder_mlp_ratio,) * self.num_hierarchies,
                (args.encoder_qkv_bias,) * self.num_hierarchies,
                (args.encoder_qk_scale,) * self.num_hierarchies
        ))
    )
        # NOTE Downstream predictor, not vit predictor for jepa
        self.predictor = models.build_predictor(
            args.predictor, self.embeddings[0],  # NOTE We have to specify which hierarchy we are using, this assumes 0 (tubesize=2)
            self.args.action_dim,
            rnn_layers=0, attentive_num_heads=args.attentive_num_heads,
            attentive_depth=args.attentive_depth
        )
        self.ema_schedulers = [
            (args.ema[0] + i*(args.ema[1]-args.ema[0])/(args.ipe*args.epochs*args.ipe_scale)
                          for i in range(int(args.ipe*args.epochs*args.ipe_scale)+1))
            for _ in range(self.num_hierarchies)
        ]
        self.ema_backbones = [copy.deepcopy(self.backbones[i]) for i in range(self.num_hierarchies)]
        self.vit_v_predictors = [
            models.VisionTransformerPredictor(tubelet_size=ts, img_size=ims, patch_size=ps, num_frames=nf,)
            for ts, ps, ims, nf in zip(
                self.tubelet_sizes, self.patch_sizes,
                (self.img_size,) * self.num_hierarchies, (self.num_frames,) * self.num_hierarchies
            )
        ]
        # NOTE There is an MLP between each hierarchy.
        # TODO I think this will need its inputs to be a fn of the 
        # tubelet size. 
        # TODO Understand the difference in input shapes between MLPs
        # and Attention based mechanisms. Something with broadcasting?
        # TODO This should be an attentive probe.
        
        # NOTE Ratio of tubelet size of i+1 to i
        self.h_tubelet_ratios = tuple(t2//t1 for t1, t2 in zip(self.tubelet_sizes, self.tubelet_sizes[1:]))
        self.attentive_h_predictors = [
            models.AttentivePooler(
                num_queries=self.num_patches_spatial[i+1],  # NOTE Next num patches
                embed_dim=self.h_tubelet_ratios[i]*self.embeddings[i],  # NOTE Predict H-4 from 2 H-2s
                out_dim=self.embeddings[i+1],
                depth=2,
            )
            for i in range(self.num_hierarchies-1)
        ]

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
    
    def forward(self, states, actions, step=None):
        """states [T, batch_size, 1, 28, 28]
        actions [T-1, batch_size]
        """
        T,B,C,H,W = states.shape
        
        ##### Tube masking
        masks_enc, masks_pred = self.tube_masking(B)
        loss = 0.
        # NOTE: Logging
        hierarchy_loss_stats = {}

        hierarchy_patches = []
        for i in range(self.num_hierarchies):
            ##### Forward target
            with torch.no_grad():
                target_patches  = self.ema_backbones[i](states, masks=masks_pred)
            ##### Forward context
            context_patches = self.backbones[i](states, masks=masks_enc)
            predicted_target_patches = self.vit_v_predictors[i](context_patches, target_patches, masks_enc, masks_pred, actions)
            loss += (h_loss := self.l2_loss(predicted_target_patches, target_patches))
            hierarchy_loss_stats[f"h{i}_reconstruction_loss"] = h_loss
            hierarchy_patches.append(predicted_target_patches)
        
        for i in range(self.num_hierarchies-1):
            predicted_next_hierarchy = self.attentive_h_predictors[i](hierarchy_patches[i])
            loss += (h_loss := self.l2_loss(predicted_next_hierarchy, hierarchy_patches[i+1]))
            hierarchy_loss_stats[f"h{i}to{i+1}_pred_loss"] = h_loss
        
        return LossInfo(total_loss=loss, diagnostics_info=hierarchy_loss_stats)
    

if __name__ == "__main__":
    states = torch.randn(16, 1, 1, 28, 28)
    actions = torch.randn(15, 1)
    hjepa = HJEPA(HJEPAConfig())
    loss = hjepa(states, actions)
    print(loss)
    