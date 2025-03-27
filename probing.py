import random
from typing import NamedTuple, List, Any, Optional
from itertools import chain
from dataclasses import dataclass
import os
import torch
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import numpy as np
import wandb

import data
from models import Prober
from configs import ConfigBase
from enums import ModelType

@dataclass
class ProbingConfig(ConfigBase):
    full_finetune: bool = False
    lr: float = 1e-3
    epochs: int = 3
    epochs_enc: int = 3
    schedule: Optional[str] = None
    prober_arch: str = ""

class ProbeResult(NamedTuple):
    model: torch.nn.Module
    average_eval_loss_unnormalized: float
    average_eval_loss_normalized: float
    eval_losses_per_step: List[float]
    plots: List[Any]


@torch.no_grad()
def probe_enc_position_visualize(
    states, pred_loc, target_loc, dataset
):

    plt.figure(dpi=200)
    pidx = 1 
    for row in range(5):
        for col in range(5):
            plt.subplot(5, 5, pidx)
            pidx += 1
            idx = pidx
            # We should always take the 2nd to last frame for visualization because this is from shadowed scope
            plt.imshow(states[idx, -2, 0].cpu(), origin="lower")
            plt.axis("off")

            pred_x_loc = dataset.unnormalize_location(pred_loc[idx][0,0])
            pred_y_loc = dataset.unnormalize_location(pred_loc[idx][0,1])

            plt.plot(
                pred_x_loc.item(),
                pred_y_loc.item(),
                marker="o",
                markersize=3,
                markeredgecolor="red",
                markerfacecolor="green",
                alpha=0.5,
            )

            target_x_loc = dataset.unnormalize_location(target_loc[idx][0,0])
            target_y_loc = dataset.unnormalize_location(target_loc[idx][0,1])
            plt.plot(
                target_x_loc.item(),
                target_y_loc.item(),
                marker="x",
                markersize=3,
                markeredgecolor="red",
                markerfacecolor="yellow",
                alpha=0.5,
            )


def probe_enc_position(
    backbone: torch.nn.Module,
    embedding: int,
    dataset,
    *,
    tubelet_size: int = 2,
    visualize: bool = False,
    prober_arch: str = "",
    quick_debug: bool = False,
    config: ProbingConfig = ProbingConfig(),
    name_suffix: str = "",
    model_type: ModelType,
    probe_model: Optional[torch.nn.Module] = None,
    cfg_name: str = "",  # Just for visualization name
):
    sf = plt.savefig
    if quick_debug:
        pass
        # Iterate thru the dataset and try to render the fucking image with the ground truths
        # for i, batch in enumerate(dataset):
        #     if i > 10: break
        #     plt.figure(dpi=200)
        #     plt.imshow(batch.states[0, 0, 0].cpu(), origin="lower")
        #     location = batch.locations[0,0,0].cpu()
        #     plt.scatter(*dataset.unnormalize_location(location), color="red", marker="x")
        #     plt.show()
        #     sf(f"visualizations/test_{i}_enc_position_visualization_{cfg_name}.png")

    test_batch = dataset[0]
    batch_size = test_batch.states.shape[0]
    num_timesteps = test_batch.states.shape[1]
    num_tubelets = num_timesteps // tubelet_size
    if model_type == ModelType.VJEPA:
        num_dots, location_dim = test_batch.locations[0, 0].shape
        prober_output_shape = (num_dots, location_dim * tubelet_size)
        # NOTE SAMI: For V-JEPA, the embedding is actually the flattened spatial patches for a tubelet.
        # For example, if each pair of frames have 49 spatial patches with a 64 embedding dimension,
        # then the input to the prober is 49*64 == 3136.
        with torch.no_grad():
            # NOTE Rearrange to make the temporal dimension the leading dimension
            e = backbone(test_batch.states.permute(1, 0, 2, 3, 4).cuda())
            # (batch_size, num_tubelets, num_spatial_patches, embedding_dim)
            e = e.view(batch_size, num_tubelets, -1)
            embedding = e.shape[-1]# 3136
    else:
        prober_output_shape = test_batch.locations[0, 0].shape

    prober = Prober(embedding, prober_arch, output_shape=prober_output_shape)
    prober = prober.cuda()

    if probe_model is not None:
        prober = probe_model
        prober = prober.cuda()

    if not config.full_finetune:
        optimizer_pred_prober = torch.optim.Adam(prober.parameters(), config.lr)
    else:
        optimizer_pred_prober = torch.optim.Adam(
            chain(prober.parameters(), backbone.parameters()),
            config.lr,
        )

    losses = []

    if quick_debug:
        config.epochs_enc = 1
    step = 0
    sample_step = 0

    ##### TRAINING #####
    if probe_model is None:
        for epoch in tqdm(range(config.epochs_enc)):
            for batch in dataset:
                target_loc = batch.locations[:, 0].cuda().float()
                if model_type == ModelType.VJEPA:
                    # NOTE SAMI: For V-JEPA we re-shape the locations to be (batch_size, num_tubelets, num_dots, 4).
                    target_loc = batch.locations.view(batch_size, num_tubelets, num_dots, location_dim * tubelet_size)
                    states = batch.states.permute(1, 0, 2, 3, 4)
                    # NOTE SAMI: For V-JEPA, we encode the entire sequence and then predict
                    # the location of the dot on each tubelet. The encoded sequence e is of shape
                    # (batch_size, num_patches_flattened, embedding_dim). For video of 18 frames, 28x28 px, and 4x4 patch size,
                    # we end up with (9x7x7, embed_dim) = (128,441,64) in this case.
                    if not config.full_finetune:
                        with torch.no_grad():
                            e = backbone(states.cuda())
                    else:
                        e = backbone(states.cuda())
                    # NOTE SAMI: Since e is flattened, we need to reshape it back to batch_size x num_tubelets x spatial_patches x embedding_dim
                    # so that we can apply the prober to tubelets of the entire frame.
                    e = e.view(batch_size, num_tubelets, -1)
                    # NOTE SAMI: Prober here should be Linear(in_features=3136, out_features=(num_dots, 4), bias=False)
                    loss = 0.0
                    # NOTE SAMI: Average loss over all the tubelets.
                    for i in range(num_tubelets):
                        pred_loc = prober(e[:, i])
                        loss += location_losses(pred_loc, target_loc[:, i])
                    loss /= num_tubelets
                else:
                    if not config.full_finetune:
                        with torch.no_grad():
                            e = backbone(batch.states[:, 0].cuda())
                    else:
                        e = backbone(batch.states[:, 0].cuda())
                    pred_loc = prober(e)
                    loss = location_losses(pred_loc, target_loc)

                losses.append(loss.mean().item())

                optimizer_pred_prober.zero_grad()
                loss.mean().backward()
                optimizer_pred_prober.step()

                if wandb.run is not None and step % 100 == 0:
                    log_dict = {
                        f"finetune_enc{name_suffix}/loss": loss.mean().item(),
                        f"finetune_enc{name_suffix}/step": step,
                        f"finetune_enc{name_suffix}/sample_step": sample_step,
                        f"finetune_enc{name_suffix}/epoch": epoch,
                    }
                    per_dot_losses = loss
                    for i, val in enumerate(per_dot_losses):
                        log_dict[f"finetune_enc{name_suffix}/loss_dot_{i}"] = val.item()
                    wandb.log(log_dict)

                step += 1
                sample_step += batch.locations.shape[0]
                if quick_debug:
                    break

    ##### EVALUATION #####
    with torch.no_grad():
        eval_losses = []
        for batch in dataset:
            if model_type == ModelType.VJEPA:
                target_loc = batch.locations.view(batch_size, num_timesteps // tubelet_size, num_dots, location_dim * tubelet_size).cuda().float()
                states = batch.states.permute(1, 0, 2, 3, 4)
                e = backbone(states.cuda())
                e = e.view(batch_size, num_tubelets, -1)
                loss = 0.0
                for i in range(num_tubelets):
                    pred_loc = prober(e[:, i])
                    target_frame_loc = target_loc[:, i]
                    loss += location_losses(pred_loc, target_frame_loc)
                loss /= num_tubelets
            else:
                target_loc = batch.locations[:, 0].cuda().float()
                e = backbone(batch.states[:, 0].cuda())
                pred_loc = prober(e)
                target_frame_loc = target_loc
                loss = location_losses(pred_loc, target_frame_loc).mean()

            eval_losses.append(loss.item())

    if visualize:
        plt.figure(dpi=200)
        plt.plot(losses)
        plt.grid()
        plt.show()
        # NOTE SAMI: Uses the fact that python still has access to the variables from the previous loop.
        probe_enc_position_visualize(
            batch.states, pred_loc, target_frame_loc, dataset)
        plt.show()
        if not os.path.exists("visualizations"):
            os.makedirs("visualizations")
        if not os.path.exists(os.path.join("visualizations", cfg_name, 'enc_position')):
            os.makedirs(os.path.join("visualizations", cfg_name, 'enc_position'))

        plt.title(f"Loss: {losses[-1]:.2f} - {cfg_name}")
        plt.savefig(os.path.join("visualizations", cfg_name, 'enc_position', f"vis_loss_{losses[-1]:.2f}.png"))

    avg_loss = np.mean(eval_losses)

    return ProbeResult(
        prober,
        average_eval_loss_unnormalized=dataset.unnormalize_mse(avg_loss),
        average_eval_loss_normalized=dataset.normalize_mse(avg_loss),
        eval_losses_per_step=eval_losses,
        plots=[],
    )


def probe_action_position_vjepa(
    backbone: torch.nn.Module,
    dataset,
    *,
    tubelet_size: int = 2,
    visualize: bool = False,
    prober_arch: str = "",
    quick_debug: bool = False,
    config: ProbingConfig = ProbingConfig(),
    name_suffix: str = "",
    within_tubelet: bool = False,
    probe_model: Optional[torch.nn.Module] = None,
) -> ProbeResult:
    test_batch = dataset[0]
    batch_size = test_batch.states.shape[0]
    num_timesteps = test_batch.states.shape[1]
    num_tubelets = num_timesteps // tubelet_size

    num_dots, action_dim = test_batch.actions[0, 0].shape
    target_action = test_batch.actions
    # NOTE: Keep actions that correspond to within tubelets, so:
    # 0->1, 2->3, 4->5, ..., or between tubelets: 1->2, 3->4, 5->6, ...
    if within_tubelet: target_action = target_action[:, 0::2, :, :]
    else:              target_action = target_action[:, 1::2, :, :]
    
    # For each frame tubelet, we flatten it (7x7x64) into a vector of size 3136, then predict 1 action.
    # In the between-tubelet case, we flatten and concat (3136x2) and predict 1 action.
    prober_output_shape = (num_dots, action_dim)
    with torch.no_grad():
        e = backbone(test_batch.states.cuda())
        e = e.view(batch_size, num_tubelets, -1)
        # NOTE: Between tubelets, we have two frames, so we double the embedding dimension.
        if within_tubelet: embedding = e.shape[-1]
        else: embedding = 2 * e.shape[-1]

    prober = Prober(embedding, prober_arch, output_shape=prober_output_shape)
    prober = prober.cuda()

    if probe_model is not None:
        prober = probe_model
        prober = prober.cuda()

    if not config.full_finetune:
        optimizer_pred_prober = torch.optim.Adam(prober.parameters(), config.lr)
    else:
        optimizer_pred_prober = torch.optim.Adam(
            chain(prober.parameters(), backbone.parameters()),
            config.lr,
        )

    losses = []

    if quick_debug:
        config.epochs_enc = 1

    step = 0
    sample_step = 0
    ##### TRAINING #####
    for epoch in tqdm(range(config.epochs_enc)):
        for batch in dataset:
            target_action = batch.actions
            if within_tubelet:  target_action = target_action[:, 0::2, :, :]
            else:               target_action = target_action[:, 1::2, :, :]

            states = batch.states.permute(1, 0, 2, 3, 4)
            if not config.full_finetune:
                with torch.no_grad():
                    e = backbone(states.cuda())
            else:
                e = backbone(states.cuda())

            e = e.view(batch_size, num_tubelets, -1)
            loss = 0.0
            if within_tubelet:
                num_actions = num_tubelets
                for i, _ in enumerate(range(0, num_timesteps, tubelet_size)):
                    # NOTE Take a tubelet and predict an action for each dot.
                    pred_action = prober(e[:, i])
                    loss += location_losses(pred_action, target_action[:, i])
                loss /= num_actions
            else: 
                # NOTE Take pairs of tubelets and predict an action for each dot.
                num_actions = num_tubelets - 1
                for i, _ in enumerate(range(1, num_timesteps-1, tubelet_size)):
                    tubelet_flattened = torch.cat([e[:, i], e[:, i-1]], dim=1)
                    pred_action = prober(tubelet_flattened)
                    loss += location_losses(pred_action, target_action[:, i])
                loss /= num_actions

            losses.append(loss.mean().item())

            optimizer_pred_prober.zero_grad()
            loss.mean().backward()
            optimizer_pred_prober.step()

            if wandb.run is not None and step % 100 == 0:
                log_dict = {
                    f"finetune_action{name_suffix}/loss": loss.mean().item(),
                    f"finetune_action{name_suffix}/step": step,
                    f"finetune_action{name_suffix}/sample_step": sample_step,
                    f"finetune_action{name_suffix}/epoch": epoch,
                }
                per_dot_losses = loss
                for i, val in enumerate(per_dot_losses):
                    log_dict[f"finetune_action{name_suffix}/loss_dot_{i}"] = val.item()
                wandb.log(log_dict)

            step += 1
            sample_step += batch.actions.shape[0]
            if quick_debug: break

    ##### EVALUATION #####
    with torch.no_grad():
        eval_losses = []
        for batch in dataset:
            if within_tubelet:
                target_action = batch.actions[:, 0::2, :, :]
            else:
                target_action = batch.actions[:, 1::2, :, :]

            states = batch.states.permute(1, 0, 2, 3, 4)
            e = backbone(states.cuda())
            e = e.view(batch_size, num_tubelets, -1)
            loss = 0.0
            if within_tubelet:
                num_actions = num_tubelets
                for i, _ in enumerate(range(0, num_timesteps-1, tubelet_size)):
                    pred_action = prober(e[:, i])
                    loss += location_losses(pred_action, target_action[:, i])
                loss /= num_actions
            else:
                num_actions = num_tubelets - 1
                for i, _ in enumerate(range(1, num_timesteps-1, tubelet_size)):
                    tubelet_flattened = torch.cat([e[:, i], e[:, i-1]], dim=1)
                    pred_action = prober(tubelet_flattened)
                    loss += location_losses(pred_action, target_action[:, i])
                loss /= num_actions

            eval_losses.append(loss.item())

    # TODO Add visualization
    

    avg_loss = np.mean(eval_losses)

    return ProbeResult(
        prober,
        average_eval_loss_unnormalized=dataset.unnormalize_mse(avg_loss),
        average_eval_loss_normalized=dataset.normalize_mse(avg_loss),
        eval_losses_per_step=eval_losses,
        plots=[],
    )


def probe_pred_position_visualize(
    model: torch.nn.Module,
    *,
    embedding: int,
    burn_in: int,
    predictor: torch.nn.Module,
    prober: torch.nn.Module,
    dataset,
    model_type: ModelType,
):
    batch = next(iter(dataset))

    burnin_states = batch.states[:, : burn_in - 1].cuda().permute(1, 0, 2, 3, 4)
    states = batch.states[:, burn_in - 1 :].cuda().permute(1, 0, 2, 3, 4)

    # drop actions of other spheres, put time first
    burnin_actions = batch.actions[:, : burn_in - 1, 0].cuda().permute(1, 0, 2)
    actions = batch.actions[:, burn_in - 1 :, 0].cuda().permute(1, 0, 2)

    if burn_in > 1:
        burnin_encodings = model(burnin_states.flatten(0, 1)).view(
            *burnin_actions.shape[:2], -1
        )
        h0 = predictor.burn_in(burnin_encodings, burnin_actions)
    else:
        h0 = None

    # For V-JEPA, we need to pass in at least two frames to get a tubelet
    if model_type == ModelType.VJEPA:
        e = model(states[0:2].cuda())
    else:
        e = model(states[0].cuda())

    pred_encs = predictor.predict_sequence(enc=e, actions=actions, h=h0)

    # for i in range(batch.actions.shape[1]):
    #     e = predictor(e, batch.actions[:, i].cuda())
    pred_loc = dataset.unnormalize_location(prober(pred_encs[-1])[:, 0])
    target_loc = dataset.unnormalize_location(batch.locations[:, -1, 0])

    fig, ax = plt.subplots(5, 5, dpi=200)
    pidx = 1
    for row in range(5):
        for col in range(5):
            # plt.subplot(5, 5, pidx)
            pidx += 1
            idx = random.randint(0, batch.states.shape[0] - 1)

            ax[row][col].imshow(batch.states[idx, 0, 0].cpu(), origin="lower")
            ax[row][col].set_axis_off()

            pred_x_loc = pred_loc[idx, 0]
            pred_y_loc = pred_loc[idx, 1]

            ax[row][col].plot(
                pred_x_loc.item(),
                pred_y_loc.item(),
                marker="o",
                markersize=2,
                markeredgecolor="red",
                markerfacecolor="green",
                alpha=0.5,
            )
            ax[row][col].plot(
                target_loc[idx, 0].item(),
                target_loc[idx, 1].item(),
                marker="x",
                markersize=2,
                markeredgecolor="red",
                markerfacecolor="yellow",
                alpha=0.5,
            )
    return fig


def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (pred - target).pow(2).sum(dim=-1).mean(dim=0)
    return mse


def probe_pred_position(
    backbone: torch.nn.Module,
    dataset,
    embedding: int,
    predictor: torch.nn.Module,
    *,
    visualize: bool = False,
    quick_debug: bool = False,
    burn_in: int = 0,
    config: ProbingConfig = ProbingConfig(),
    name_suffix: str = "",
    probe_model: Optional[torch.nn.Module] = None,
):
   
    # NOTE SAMI: Autoregressively rollout future states from first state and all actions
    # and then take loss on each predicted state with its ground truth.
    # VJEPA Can't really do this because our predictor is self-predictive and is not a dynamics model.
    # What we might be able to do instead is to take a tubelets (2 frames) and then predict the action from those tubelets.
    # Or also, take the tubelets (2 frames) and predict the two states (locations) from those tubelets. if quick_debug:
    config.epochs = 1
    test_batch = next(iter(dataset))

    prober_output_shape = test_batch.locations[0, 0].shape
    prober = Prober(embedding, config.prober_arch, output_shape=prober_output_shape)
    prober = prober.cuda()

    if probe_model is not None:
        prober = probe_model
        prober = prober.cuda()

    if not config.full_finetune:
        optimizer_pred_prober = torch.optim.Adam(prober.parameters(), config.lr)
    else:
        optimizer_pred_prober = torch.optim.Adam(
            chain(prober.parameters(), backbone.parameters(), predictor.parameters()),
            config.lr,
        )

    sample_step = 0
    step = 0

    for epoch in tqdm(range(config.epochs)):
        for batch in dataset:
            # put time first
            burnin_states = batch.states[:, : burn_in - 1].cuda().permute(1, 0, 2, 3, 4)
            states = batch.states[:, burn_in - 1 :].cuda().permute(1, 0, 2, 3, 4)

            # drop actions of other spheres, put time first
            burnin_actions = batch.actions[:, : burn_in - 1, 0].cuda().permute(1, 0, 2)
            actions = batch.actions[:, burn_in - 1 :, 0].cuda().permute(1, 0, 2)

            if burn_in > 1:
                burnin_encodings = backbone(burnin_states.flatten(0, 1)).view(
                    *burnin_actions.shape[:2], -1
                )
                h0 = predictor.burn_in(burnin_encodings, burnin_actions)
            else:
                h0 = None
            # For V-JEPA, we need to pass in at least two frames to get a tubelet
            e = backbone(states[0])


            pred_encs = predictor.predict_sequence(enc=e, actions=actions, h=h0)
            if not config.full_finetune:
                pred_encs = pred_encs.detach()

            all_encs = torch.cat([e.unsqueeze(0), pred_encs], dim=0)
            pred_locs = torch.stack([prober(x) for x in all_encs], dim=1)

            losses = location_losses(
                pred_locs, batch.locations[:, burn_in - 1 :].cuda()
            )

            loss = losses.mean()
            optimizer_pred_prober.zero_grad()
            loss.backward()
            optimizer_pred_prober.step()

            if wandb.run is not None and step % 100 == 0:
                log_dict = {
                    f"finetune_pred{name_suffix}/loss": loss.item(),
                    f"finetune_pred{name_suffix}/step": step,
                    f"finetune_pred{name_suffix}/sample_step": sample_step,
                    f"finetune_pred{name_suffix}/epoch": epoch,
                }
                per_dot_losses = losses.mean(dim=0)
                for i, val in enumerate(per_dot_losses):
                    log_dict[f"finetune_pred{name_suffix}/loss_dot_{i}"] = val.item()

                wandb.log(log_dict)

            step += 1
            sample_step += states.shape[0]

            if quick_debug:
                break

    with torch.no_grad():
        eval_losses = []
        for batch in dataset:
            # put time first
            burnin_states = batch.states[:, : burn_in - 1].cuda().permute(1, 0, 2, 3, 4)
            states = batch.states[:, burn_in - 1 :].cuda().permute(1, 0, 2, 3, 4)

            # drop actions of other spheres, put time first
            burnin_actions = batch.actions[:, : burn_in - 1, 0].cuda().permute(1, 0, 2)
            actions = batch.actions[:, burn_in - 1 :, 0].cuda().permute(1, 0, 2)

            if burn_in > 1:
                burnin_encodings = backbone(burnin_states.flatten(0, 1)).view(
                    *burnin_actions.shape[:2], -1
                )
                h0 = predictor.burn_in(burnin_encodings, burnin_actions)
            else:
                h0 = None

            e = backbone(states[0])
            pred_encs = predictor.predict_sequence(
                enc=e, actions=actions, h=h0
            ).detach()
            all_encs = torch.cat([e.unsqueeze(0), pred_encs], dim=0)

            pred_locs = torch.stack([prober(x) for x in all_encs], dim=1)

            losses = location_losses(
                pred_locs, batch.locations[:, burn_in - 1 :].cuda()
            )

            eval_losses.append(losses)

            if quick_debug:
                break

    
        losses_t = torch.stack(eval_losses, dim=0).mean(dim=0)
        unnormalized_losses_t = dataset.unnormalize_mse(losses_t)
        normalized_losses_t = dataset.normalize_mse(losses_t)
    if visualize:
        fig = probe_pred_position_visualize(
            backbone,
            dataset=dataset,
            embedding=embedding,
            burn_in=burn_in,
            predictor=predictor,
            prober=prober,
        )
    else:
        fig = None

    return ProbeResult(prober,
        average_eval_loss_unnormalized=unnormalized_losses_t.mean().item(),
        average_eval_loss_normalized=normalized_losses_t.mean().item(),
        eval_losses_per_step=losses_t,
        plots=[fig],
    )


class ProbeMPCResult(NamedTuple):
    average_diff: float
    figures: List[Any]


def normalize_actions(actions):
    actions_n = actions.clone()
    actions_n[..., :2] = (
        actions[..., :2] / actions[..., :2].norm(dim=-1).unsqueeze(-1).detach()
    )
    actions_n[..., -1] = actions[..., -1].clamp(min=0, max=4)
    return actions_n


def probe_mpc(
    backbone: torch.nn.Module,
    *,
    embedding: int,
    predictor: torch.nn.Module,
    prober: torch.nn.Module,
    plan_size: int = 17,
    n_iters: int = 20,
):
    prober.eval()
    figs = []
    diffs = []
    for i in range(n_iters):
        state1, location1 = data.ContinuousMotionDataset.generate_state()
        state2, location2 = data.ContinuousMotionDataset.generate_state()

        # plt.subplot(1, 2, 1)
        # plt.imshow(state1[0], origin='lower')
        # plt.subplot(1, 2, 2)
        # plt.imshow(state2[0], origin=''lower)

        enc1 = backbone(state1.unsqueeze(0).cuda())
        # enc2 = backbone(state2.unsqueeze(0).cuda())

        directions = torch.rand((plan_size, 2), device="cuda") * 2 - 1
        speeds = torch.rand((plan_size, 1), device="cuda") * 4
        actions = normalize_actions(torch.cat([directions, speeds], dim=-1))
        actions.requires_grad = True

        opt = torch.optim.Adam((actions,), lr=0.1)

        losses = []
        for _ in range(100):
            current_enc = enc1.detach()
            actions_n = normalize_actions(actions)
            actions_2 = actions_n[:, :2] * actions[:, 2].unsqueeze(-1)

            pred_encs = predictor.predict_sequence(
                enc=current_enc, actions=actions_2.unsqueeze(1).cuda()
            )
            # for i in range(actions.shape[0]):
            #     # t = torch.concat([current_enc, actions_2[i].unsqueeze(0)], dim=1)
            #     # current_enc = predictor.model(t)
            #     current_enc = predictor(current_enc, actions_2[i].unsqueeze(0))

            pred_loc = prober(pred_encs[-1])
            #     target_loc = prober(enc2)
            target_loc = location2.cuda().float().unsqueeze(0)
            #     diff = torch.nn.functional.mse_loss(current_enc, enc2.detach())
            diff = torch.nn.functional.mse_loss(pred_loc, target_loc.detach())
            #     diff = -1 * current_enc[0].T @ enc2[0].detach()
            #     print(diff.shape, current_enc.shape)
            #     print(pred_loc, target_loc)
            opt.zero_grad()
            diff.backward()
            opt.step()
            losses.append(diff.item())

        actions_n = normalize_actions(actions)
        actions_2 = actions_n[:, :2] * actions[:, 2].unsqueeze(-1)

        seq = data.ContinuousMotionDataset.generate_transitions(
            state1, location1, actions_2.cpu()
        )

        # fig, ax = plt.subplots(1, 5, dpi=200)

        fig = plt.figure(dpi=200)
        gs = fig.add_gridspec(2, 4)
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[1, 2])
        ax4 = fig.add_subplot(gs[1, 3])

        ax0.plot(losses)
        ax0.set_title("loss")
        ax0.grid(True)

        ax1.imshow(state1[0], origin="lower")
        ax1.set_xlabel(f"loc={list(map(lambda x: round(x, 1), location1.tolist()))}")
        ax1.set_title("start")

        ax2.imshow(state2[0], origin="lower")
        ax2.set_xlabel(f"loc={list(map(lambda x: round(x, 1), location2.tolist()))}")
        ax2.set_title("target")

        ax3.imshow(seq.states[-1, 0].detach().cpu(), origin="lower")
        ax3.set_xlabel(
            f"loc={list(map(lambda x: round(x, 1), seq.locations[-1].tolist()))}"
        )
        ax3.set_title("reached")

        ax4.set_title("path")
        ax4.scatter(
            seq.locations[:, 0].detach().cpu(),
            seq.locations[:, 1].detach().cpu(),
            c=range(seq.locations.shape[0]),
        )
        ax4.plot(location2[0], location2[1], marker="x", markersize=10, c="r")
        ax4.set_xlim(0, 28)
        ax4.set_ylim(0, 28)
        ax4.grid(visible=True)
        ax4.set_aspect("equal")

        fig.set_tight_layout(True)

        figs.append(fig)

        diff = (seq.locations[-1] - location2).pow(2).mean().item()
        diffs.append(diff)
    return ProbeMPCResult(np.mean(diffs), figs)
