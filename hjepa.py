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
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # -- video data
        self.num_frames = args.num_frames
        self.img_size = args.img_size
        # -- hierarchy
        # Validate that we have the same number of hierarchies for each
        assert len(args.patch_sizes) == len(args.tubelet_sizes) == len(args.masking_ratios)
        assert tuple(sorted(args.tubelet_sizes)) == tuple(args.tubelet_sizes)
        self.num_hierarchies = len(args.patch_sizes)
        self.masking_ratios = sorted(args.masking_ratios)
        self.patch_sizes: tuple[int, ...] = args.patch_sizes
        self.tubelet_sizes: tuple[int, ...] = args.tubelet_sizes
        self.nums_patches_spatial: tuple[int, ...] = tuple(args.img_size // p for p in self.patch_sizes)
        self.nums_keep_spatial: tuple[int, ...] = tuple(
            int(self.nums_patches_spatial[i] * (1 - self.masking_ratios[i]))
            for i in range(self.num_hierarchies)
        )
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
                self.patch_sizes,
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
        for v in self.vit_v_predictors: v = v.cuda()
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
                num_queries=self.nums_patches_spatial[i+1],  # NOTE Next num patches
                out_dim=self.embeddings[i+1],
                depth=2,
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

    def sample_mask(self, hierarchy_idx: int):
        mask = np.hstack([
            np.zeros(self.nums_patches_spatial[hierarchy_idx] - self.nums_keep_spatial[hierarchy_idx]),
            np.ones(self.nums_keep_spatial[hierarchy_idx]),
        ])
        np.random.shuffle(mask)
        mask = torch.tensor(np.tile(mask, (self.num_frames, 1)))
        mask = mask.flatten()
        mask_p = torch.argwhere(mask == 0).squeeze()
        mask_e = torch.nonzero(mask).squeeze()
        return mask_e, mask_p
    
    def tube_masking(self, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        collated_masks_pred, collated_masks_enc = [], []
        for _ in range(batch_size):
            masks_e, masks_p = zip(*[self.sample_mask(i) for i in range(self.num_hierarchies)])
            masks_e, masks_p = torch.stack(masks_e).to(self.device), torch.stack(masks_p).to(self.device)
            collated_masks_enc.append(masks_e)
            collated_masks_pred.append(masks_p)

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

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
            masks_pred_i, masks_enc_i = masks_pred[:, i], masks_enc[:, i]
            ##### Forward target
            with torch.no_grad():
                target_patches  = self.ema_backbones[i](states, masks=masks_pred_i)
            ##### Forward context
            context_patches = self.backbones[i](states, masks=masks_enc_i)
            predicted_target_patches = self.vit_v_predictors[i](context_patches, target_patches, masks_enc_i, masks_pred_i, actions)
            loss += (h_loss := self.l2_loss(predicted_target_patches, target_patches))
            hierarchy_loss_stats[f"h{i}_reconstruction_loss"] = h_loss
            hierarchy_patches.append(predicted_target_patches)
        
        # Next-hierarchy distillation
        for i in range(self.num_hierarchies-1):
            predicted_next_hierarchy = self.attentive_h_predictors[i](hierarchy_patches[i])
            loss += (h_loss := self.l2_loss(predicted_next_hierarchy, hierarchy_patches[i+1]))
            hierarchy_loss_stats[f"h{i}to{i+1}_pred_loss"] = h_loss
        
        return LossInfo(total_loss=loss, diagnostics_info=hierarchy_loss_stats)
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = torch.randn(16, 1, 1, 28, 28).to(device)
    actions = torch.randn(15, 16).to(device)
    hjepa = HJEPA(HJEPAConfig()).cuda()
    loss = hjepa(states, actions)
    print(loss)
    