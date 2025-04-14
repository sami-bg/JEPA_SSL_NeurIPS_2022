import data
from typing import Optional
import os
from dataclasses import dataclass, field
import dataclasses
from pathlib import Path
import random

import torch
import numpy as np
from tqdm.auto import tqdm
import wandb
from omegaconf import OmegaConf, MISSING
from matplotlib import pyplot as plt
import matplotlib
from torch import nn

from configs import ConfigBase
from rssm import RSSMConfig, RSSMPredMultistep
from simclr import SimCLRConfig, SimCLR
from vicreg import VICRegConfig, VICRegPredMultistep
from lars import LARS, exclude_bias_and_norm, adjust_learning_rate
import probing
from vjepa import VJEPAConfig, VJEPA
from enums import ModelType, DatasetType

os.environ['WANDB_DISABLED'] = "true"


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


OmegaConf.register_new_resolver("eval", eval)

@dataclass
class TrainConfig(ConfigBase):
    model_type: ModelType = MISSING
    n_steps: int = 18
    val_n_steps: int = 18
    dataset_size: int = 10000
    dataset_noise: float = 0.0
    val_dataset_size: int = 10000
    wandb: bool = True
    run_name: Optional[str] = None
    run_group: Optional[str] = None
    output_path: Optional[str] = None
    safe_every_n_epochs: int = 10
    eval_mpcs: int = 20
    quick_debug: bool = False
    seed: int = 42
    load_checkpoint_path: Optional[str] = None
    load_probing_checkpoint_path: Optional[str] = None
    eval_only: bool = False
    probe_mpc: bool = False
    dataset_static_noise: float = 0.0
    dataset_structured_noise: bool = False
    dataset_structured_noise_path: Optional[str] = "/tmp/cifar10"
    dataset_batch_size: int = 128
    dataset_static_noise_speed: float = 0.0
    dataset_dot_std: float = 1.3
    dataset_normalize: bool = False
    vicreg: VICRegConfig = field(default_factory=VICRegConfig)
    rssm: RSSMConfig = field(default_factory=RSSMConfig)
    simclr: SimCLRConfig = field(default_factory=SimCLRConfig)
    vjepa: VJEPAConfig = field(default_factory=VJEPAConfig)
    eval_at_the_end_only: bool = False
    dataset_type: DatasetType = DatasetType.Single
    cfg_name: str = ""
    probing_cfg: probing.ProbingConfig = field(default_factory=probing.ProbingConfig)


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        if self.config.model_type == ModelType.RSSM:
            self.model_config = self.config.rssm
            self.pred_ms = RSSMPredMultistep(config.rssm)
            self.pred_ms = self.pred_ms.cuda()
        elif self.config.model_type == ModelType.VICReg:
            self.model_config = self.config.vicreg
            self.pred_ms = VICRegPredMultistep(self.config.vicreg)
            self.pred_ms = self.pred_ms.cuda()
        elif self.config.model_type == ModelType.SimCLR:
            self.model_config = self.config.simclr
            self.pred_ms = SimCLR(config.simclr)
            self.pred_ms = self.pred_ms.cuda()
        elif self.config.model_type == ModelType.VJEPA:
            self.model_config = self.config.vjepa
            self.pred_ms = VJEPA(self.config.vjepa)
            self.pred_ms = self.pred_ms.cuda()
        else:
            raise ValueError(f"No valid model type : {self.config.model_type}")
        if config.wandb:
            wandb.init(
                project="HJEPA-debug",
                name=config.run_name,
                group=config.run_group,
                config=dataclasses.asdict(config),
            )
        print(f'Config:\n{OmegaConf.to_yaml(config)}')
        seed_everything(config.seed)

        self.sample_step = 0
        self.epoch = 0
        self.step = 0

        self.save_config()
        self.init_dataset()
        self.init_optimizer()

        load_result = self.maybe_load_checkpoint()
        if (
            config.eval_only
            and not config.probing_cfg.full_finetune
            and not load_result
        ):
            print("WARN: probing a random network. Is that intentional?")

        self.n_parameters = sum(
            p.numel() for p in self.pred_ms.parameters() if p.requires_grad
        )
        print("number of params:", self.n_parameters)
        if config.wandb:
            wandb.run.summary["n_params"] = self.n_parameters
            wandb.run.summary["actual_repr_size"] = self.pred_ms.embedding

    def init_optimizer(self):
        if self.config.model_type in [ModelType.VICReg, ModelType.SimCLR, ModelType.VJEPA]:
            self.optimizer = LARS(
                self.pred_ms.parameters(),
                lr=0,
                weight_decay=1e-6,
                weight_decay_filter=exclude_bias_and_norm,
                lars_adaptation_filter=exclude_bias_and_norm,
            )
        elif self.config.model_type == ModelType.RSSM:
            self.optimizer = torch.optim.Adam(
                self.pred_ms.parameters(),
                lr=self.model_config.learning_rate,
                eps=self.model_config.rssm_adam_epsilon,
            )

    def init_dataset(self):
        if self.config.dataset_type == DatasetType.Single:
            assert self.pred_ms.args.channels == 1, (
                "encoder and dataset type are incompatible,"
                f"single dot datset provides one channel, while {self.pred_ms.args.channels} were expected"
            )
            self.ds = data.ContinuousMotionDataset(
                self.config.val_dataset_size // self.config.dataset_batch_size,
                batch_size=self.config.dataset_batch_size,
                n_steps=self.config.n_steps + self.pred_ms.args.rnn_burnin - 1,
                noise=self.config.dataset_noise,
                static_noise=self.config.dataset_static_noise,
                static_noise_speed=self.config.dataset_static_noise_speed,
                structured_noise=self.config.dataset_structured_noise,
                structured_dataset_path=self.config.dataset_structured_noise_path,
                std=self.config.dataset_dot_std,
                normalize=self.config.dataset_normalize,
                device=torch.device("cuda"),
                train=True,
            )
            self.val_ds = data.ContinuousMotionDataset(
                self.config.val_dataset_size // self.config.dataset_batch_size,
                batch_size=self.config.dataset_batch_size,
                n_steps=self.config.val_n_steps + self.pred_ms.args.rnn_burnin - 1,
                noise=self.config.dataset_noise,
                static_noise=self.config.dataset_static_noise,
                static_noise_speed=self.config.dataset_static_noise_speed,
                structured_noise=self.config.dataset_structured_noise,
                structured_dataset_path=self.config.dataset_structured_noise_path,
                std=self.config.dataset_dot_std,
                normalize=self.config.dataset_normalize,
                device=torch.device("cuda"),
                train=False,
            )
        elif self.config.dataset_type == DatasetType.Multiple:
            sum_image = self.pred_ms.args.channels == 1
            self.ds = data.create_three_datasets(
                self.config.val_dataset_size // self.config.dataset_batch_size,
                batch_size=self.config.dataset_batch_size,
                n_steps=self.config.n_steps + self.pred_ms.args.rnn_burnin - 1,
                noise=self.config.dataset_noise,
                static_noise=self.config.dataset_static_noise,
                static_noise_speed=self.config.dataset_static_noise_speed,
                structured_noise=self.config.dataset_structured_noise,
                structured_dataset_path=self.config.dataset_structured_noise_path,
                std=self.config.dataset_dot_std,
                normalize=self.config.dataset_normalize,
                sum_image=sum_image,
                device=torch.device("cuda"),
                train=True,
            )
            self.val_ds = data.create_three_datasets(
                self.config.val_dataset_size // self.config.dataset_batch_size,
                batch_size=self.config.dataset_batch_size,
                n_steps=self.config.val_n_steps + self.pred_ms.args.rnn_burnin - 1,
                noise=self.config.dataset_noise,
                static_noise=self.config.dataset_static_noise,
                static_noise_speed=self.config.dataset_static_noise_speed,
                structured_noise=self.config.dataset_structured_noise,
                structured_dataset_path=self.config.dataset_structured_noise_path,
                std=self.config.dataset_dot_std,
                normalize=self.config.dataset_normalize,
                sum_image=sum_image,
                device=torch.device("cuda"),
                train=False,
            )
        else:
            raise NotImplementedError(
                f"dataset type {self.config.dataset_type} is not supported"
            )

    def maybe_load_checkpoint(self):
        if self.config.load_checkpoint_path is not None:
            checkpoint = torch.load(self.config.load_checkpoint_path, weights_only=False)
            self.pred_ms.load_state_dict(checkpoint["model_state_dict"])
            self.epoch = checkpoint.get("epoch", 0)
            self.sample_step = checkpoint.get("sample_step", 0)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            return True
        return False

    def save_config(self):
        if self.config.output_path is not None:
            os.makedirs(self.config.output_path, exist_ok=True)
            p = Path(self.config.output_path) / "config.yaml"
            with p.open("w") as f:
                OmegaConf.save(config=self.config, f=f)
                print("saved config")

    def train(self):
        self.save_checkpoint()

        if not self.config.eval_at_the_end_only or self.config.quick_debug:
            self.validate()

        for epoch in tqdm(range(1, self.pred_ms.args.epochs + 1)):
            self.epoch = epoch
            for step, batch in tqdm(enumerate(self.ds, start=epoch * len(self.ds))):
                # move to cuda and shuffle batch and time
                s = batch.states.cuda().permute(1, 0, 2, 3, 4)
                a = batch.actions.cuda().permute(1, 0, 2, 3)
                a = a[:, :, 0]  # we only get the actions of the first dot

                if self.config.model_type == ModelType.RSSM:
                    lr = self.model_config.learning_rate
                    if epoch > 36:
                        lr = self.model_config.learning_rate / 10
                        optimizer = torch.optim.Adam(
                            self.pred_ms.parameters(),
                            lr=self.model_config.learning_rate / 10,
                            eps=self.model_config.rssm_adam_epsilon,
                        )
                elif self.config.model_type in [ModelType.VICReg, ModelType.SimCLR, ModelType.VJEPA]:
                    lr = adjust_learning_rate(
                        self.model_config, self.optimizer, self.ds, step
                    )

                self.sample_step += s.shape[1]
                self.step = step

                self.optimizer.zero_grad()
                loss_info = self.pred_ms.forward(s, a, step=self.step)
                loss_info.total_loss.backward()
                if self.config.model_type == ModelType.RSSM:
                    nn.utils.clip_grad_norm_(
                        self.pred_ms.parameters(), 1000, norm_type=2
                    )
                self.optimizer.step()

                if step % 100 == 0 and wandb.run is not None:
                    wandb.log(
                        {
                            "sample_step": self.sample_step,
                            "loss": loss_info.total_loss.item(),
                            "learning_rate": lr,
                            "custom_step": step,
                        },
                        commit=False,
                    )
                    if loss_info.diagnostics_info is not None:
                        wandb.log(
                            loss_info.diagnostics_info.build_log_dict(),
                        )
                if self.config.quick_debug:
                    return  # just finish after one step

            if (
                self.epoch % 20 == 0 and not self.config.eval_at_the_end_only
            ) or self.epoch >= self.model_config.epochs:
                self.save_checkpoint()
                self.validate()

    def validate(self):
        if not self.config.probing_cfg.full_finetune:
            self.pred_ms.eval()

        log_dict = {
            "epoch": self.epoch,
            "sample_step": self.sample_step,
        }
        save_dict = {
            "epoch": self.epoch,
            "sample_step": self.sample_step,
            "probe_enc_model": None,
            "probe_pred_model": None,
            "probe_mpc_model": None,
            "probe_action_model_within_tubelet": None,
            "probe_action_model_between_tubelet": None,
        }

        if self.config.load_probing_checkpoint_path is not None:
            checkpoint = torch.load(self.config.load_probing_checkpoint_path)
            save_dict["probe_enc_model"] = checkpoint.get("probe_enc_model", None)
            save_dict["probe_pred_model"] = checkpoint.get("probe_pred_model", None)
            save_dict["probe_mpc_model"] = checkpoint.get("probe_mpc_model", None)
            save_dict[f"probe_action_model_within_tubelet"] = checkpoint.get("probe_action_model_within_tubelet", None)
            save_dict[f"probe_action_model_between_tubelet"] = checkpoint.get("probe_action_model_between_tubelet", None)
            if None in save_dict.values():
                print(f"WARNING: Some models were not loaded from checkpoint: {checkpoint.keys()}, {save_dict=}")

        probing_enc_result = probing.probe_enc_position(
            backbone=self.pred_ms.backbone,
            embedding=self.pred_ms.embedding,
            dataset=self.val_ds,
            tubelet_size=self.pred_ms.args.tubelet_size,
            quick_debug=self.config.quick_debug,
            config=self.config.probing_cfg,
            name_suffix=f"_{self.epoch}",
            model_type=self.config.model_type,
            visualize=self.config.model_type == ModelType.VJEPA,
            probe_model=save_dict["probe_enc_model"],
            cfg_name=self.config.cfg_name
        )
        save_dict["probe_enc_model"] = probing_enc_result.model

        # if self.config.model_type == ModelType.VJEPA:
        if False:
            for within_tubelet in [True, False]:
                suffix = "_within_tubelet" if within_tubelet else "_between_tubelet"
                probing_action_result = probing.probe_action_position_vjepa(
                    backbone=self.pred_ms.backbone,
                    dataset=self.val_ds,
                    tubelet_size=self.pred_ms.args.tubelet_size,
                    quick_debug=self.config.quick_debug,
                    config=self.config.probing_cfg,
                    within_tubelet=within_tubelet,
                    visualize=True,
                    name_suffix=f"_{self.epoch}{suffix}",
                    probe_model=save_dict[f"probe_action_model{suffix}"]
                )
                save_dict[f"probe_action_model{suffix}"] = probing_action_result.model
                # TODO Make each fn return a ProbingResult and then log as per pred_position?
                log_dict[f"avg_eval_action_loss_unnormalized{suffix}"] = probing_action_result.average_eval_loss_unnormalized
                log_dict[f"avg_eval_action_loss_normalized{suffix}"] = probing_action_result.average_eval_loss_normalized

        
        log_dict["avg_eval_enc_loss_unnormalized"] = probing_enc_result.average_eval_loss_unnormalized
        log_dict["avg_eval_enc_loss_unnormalized_rmse"] = np.sqrt(probing_enc_result.average_eval_loss_unnormalized)
        log_dict["avg_eval_enc_loss_normalized"] = probing_enc_result.average_eval_loss_normalized
        log_dict["avg_eval_enc_loss_normalized_rmse"] = np.sqrt(probing_enc_result.average_eval_loss_normalized)


        if self.config.model_type != ModelType.VJEPA:
            # NOTE SAMI: VJEPA can't do this because our predictor is self-predictive and is not a dynamics model.
            probing_result = probing.probe_pred_position(
                self.pred_ms.backbone,
                dataset=self.val_ds,
                embedding=self.pred_ms.embedding,
                tubelet_size=self.pred_ms.args.tubelet_size,
                predictor=self.pred_ms.predictor,
                visualize=False,
                quick_debug=self.config.quick_debug,
                burn_in=self.pred_ms.args.rnn_burnin,
                config=self.config.probing_cfg,
                name_suffix=f"_{self.epoch}",
                probe_model=save_dict["probe_pred_model"]
            )
            save_dict["probe_pred_model"] = probing_result.model
            log_dict["avg_eval_rollout_loss_unnormalized"] = probing_result.average_eval_loss_unnormalized
            log_dict["avg_eval_rollout_loss_unnormalized_rmse"] = np.sqrt(probing_result.average_eval_loss_unnormalized)
            log_dict["avg_eval_rollout_loss_normalized"] = probing_result.average_eval_loss_normalized
            log_dict["avg_eval_rollout_loss_normalized_rmse"] = np.sqrt(probing_result.average_eval_loss_normalized)

            for i in range(probing_result.eval_losses_per_step.shape[0]):
                for j in range(probing_result.eval_losses_per_step.shape[1]):
                    log_dict[f"eval/loss_{i}_{j}"] = probing_result.eval_losses_per_step[
                        i, j
                    ].item()
                    log_dict[f"eval/rollout_loss_{i}_{j}_rmse"] = np.sqrt(
                        probing_result.eval_losses_per_step[i, j].item()
                    )
            for j in range(probing_result.eval_losses_per_step.shape[1]):
                log_dict[f"eval/rollout_loss_{j}"] = (
                    probing_result.eval_losses_per_step[:, j].mean().item()
                )
                log_dict[f"eval/rollout_loss_{j}_rmse"] = np.sqrt(
                    probing_result.eval_losses_per_step[:, j].mean().item()
                )

            if self.config.probe_mpc:
                probe_mpc_result = probing.probe_mpc(
                    self.pred_ms.backbone,
                    embedding=self.pred_ms.embedding,
                    predictor=self.pred_ms.predictor,
                    prober=probing_result.model,
                    plan_size=self.config.val_n_steps,
                )
                save_dict["probe_mpc_model"] = probe_mpc_result.model
                log_dict["average_mpc_mse"] = probe_mpc_result.average_diff
                for i in range(len(probe_mpc_result.figures)):
                    log_dict[f"mpc_{i}"] = probe_mpc_result.figures[i]

        log_dict["custom_step"] = self.step

        if self.config.wandb:
            wandb.log(log_dict)
            wandb.run.summary.update(log_dict)

        for v in log_dict.values():
            if isinstance(v, matplotlib.figure.Figure):
                plt.close(v)

        self.pred_ms.train()
        if self.config.output_path is not None:
            os.makedirs(self.config.output_path, exist_ok=True)
            torch.save(save_dict, os.path.join(self.config.output_path, f"probing_epoch={self.epoch}_sample_step={self.sample_step}.ckpt"))

        return log_dict

    def save_checkpoint(self):
        if self.config.output_path is not None:
            os.makedirs(self.config.output_path, exist_ok=True)
            torch.save(
                {
                    "epoch": self.epoch,
                    "sample_step": self.sample_step,
                    "model_state_dict": self.pred_ms.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                os.path.join(
                    self.config.output_path,
                    f"epoch={self.epoch}_sample_step={self.sample_step}.ckpt",
                ),
            )


def main(config: TrainConfig):
    torch.set_num_threads(1)
    trainer = Trainer(config)
    if not config.eval_only:
        trainer.train()
    else:
        trainer.validate()


if __name__ == "__main__":
    # import sys
    # sys.argv[1:] = [
    #     "--config", "reproduce_configs/vjepa/fixed_structured/sweep_fixed_structured.(0.50).vjepa.yaml"
    # ]
    cfg = TrainConfig.parse_from_command_line()
    print(OmegaConf.to_yaml(cfg))
    main(cfg)