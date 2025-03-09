import math
from typing import List, Optional

import torch
from torch import nn
import numpy as np

from torch.nn import functional as F
from resnet import resnet18, resnet18ID
from pos_embed import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed
from utils import apply_masks, trunc_normal_, repeat_interleave_batch
import resnet

ResNet18 = resnet18
ResNet18ID = resnet18ID


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, 128)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        return out


class MeNet5(nn.Module):
    def __init__(
        self, output_dim: int = 64, input_channels: int = 1, width_factor: int = 1
    ):
        super().__init__()
        self.width_factor = width_factor
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                input_channels, 16 * width_factor, kernel_size=5, stride=2, padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16 * width_factor),
            nn.Conv2d(
                16 * width_factor, 32 * width_factor, kernel_size=5, stride=2, padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32 * width_factor),
            nn.Conv2d(
                32 * width_factor, 32 * width_factor, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32 * width_factor),
            nn.AvgPool2d(2, stride=2),
        )
        self.fc = nn.Linear(9 * 32 * width_factor, output_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class ResizeConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        scale_factor,
        mode="nearest",
        groups=1,
        bias=False,
        padding=1,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class MeNet5Decoder(nn.Module):
    def __init__(self, embedding_size, output_channels: int = 1, width_factor: int = 1):
        super().__init__()
        self.width_factor = width_factor
        self.layers = nn.Sequential(
            ResizeConv2d(
                32 * width_factor,
                32 * width_factor,
                kernel_size=3,
                scale_factor=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32 * width_factor),
            ResizeConv2d(
                32 * width_factor,
                16 * width_factor,
                kernel_size=5,
                scale_factor=3,
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16 * width_factor),
            ResizeConv2d(
                16 * width_factor,
                output_channels,
                kernel_size=5,
                scale_factor=3,
                padding=2,
            ),
        )
        self.fc = nn.Linear(2 * embedding_size, 32 * 3 * 3 * self.width_factor)

    def forward(self, x: torch.Tensor, belief: torch.Tensor):
        x = torch.cat([x, belief], dim=1)
        x = self.fc(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1, 3, 3)
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # 6 by 6, undo avg pool
        x = self.layers(x)
        x = F.interpolate(x, size=(28, 28), mode="bilinear")  # 27 by 27 to 28 by 28
        return x


class Canonical(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        res = int(np.sqrt(output_dim / 64))
        assert (
            res * res * 64 == output_dim
        ), "canonical backbone resolution error: cant fit desired output_dim"

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((res, res)),
        )

    def forward(self, x):
        return self.backbone(x).flatten(1)


class MLPNet(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, output_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        out = x.flatten(1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class PassThrough(nn.Module):
    def forward(self, x):
        return x.view(*x.shape[:-3], -1)


class PixelPredictorConv(torch.nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )
        self.proj = nn.Linear(action_dim, 64)

    def forward(self, x, a):
        original_shape = x.shape
        if x.shape[-1] == 784:
            x = x.view(*x.shape[:-1], 1, 28, 28)
        a_proj = self.proj(a)
        e = self.enc(x)
        e = e + a_proj.view(*a_proj.shape, 1, 1)
        d = self.dec(e)
        return d.view(*original_shape)

    def predict_sequence(self, h: torch.Tensor, actions: torch.Tensor):
        outputs = []
        for i in range(len(actions)):
            h = self(h, actions[i])
            outputs.append(h)
        return outputs


class VAEDecoder(torch.nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.k1, self.k2, self.k3, self.k4 = (
            (3, 3),
            (3, 3),
            (3, 3),
            (3, 3),
        )  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
        )  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        )  # 2d padding

        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=self.k4,
                stride=self.s4,
                padding=self.pd4,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16, momentum=0.01),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=self.k3,
                stride=self.s3,
                padding=self.pd3,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8, momentum=0.01),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=8,
                kernel_size=self.k2,
                stride=self.s2,
                padding=self.pd2,
            ),
            nn.ReLU(),  # y = (y1, y2, y3) \in [0 ,1]^3
            nn.BatchNorm2d(8, momentum=0.01),
            nn.Conv2d(8, out_channels=1, kernel_size=3, padding=1),
        )

    #         self.fc1 = nn.Linear(embedding_size, embedding_size)

    def forward(self, z):
        x = z.view(-1, 32, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(28, 28), mode="bilinear")
        return x


class PixelEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        # self.dec = VAEDecoder(512)
        # self.proj = nn.Linear(action_dim, 512)

    def forward(self, x):
        # a_proj = self.proj(a)
        return self.enc(x).flatten(1)


def build_projector(arch: str, embedding: int):
    if arch == "id":
        return nn.Identity(), embedding
    else:
        f = [embedding] + list(map(int, arch.split("-")))
        return build_mlp(f), f[-1]


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1], bias=False))
    return nn.Sequential(*layers)


class Projector(torch.nn.Module):
    def __init__(self, arch: str, embedding: int, random: bool = False):
        super().__init__()

        self.arch = arch
        self.embedding = embedding
        self.random = random

        self.model, self.output_dim = build_projector(arch, embedding)

        if self.random:
            for param in self.parameters():
                param.requires_grad = False

    def maybe_reinit(self):
        if self.random and self.arch != "id":
            for param in self.parameters():
                torch.nn.init.xavier_uniform_(param)
                print("initialized")

    def forward(self, x: torch.Tensor):
        return self.model(x)


def build_backbone(
    arch: str,
    embedding_size: int,
    backbone_mlp: str,
    backbone_width_factor: int,
    channels: int = 1,
    **kwargs,
):
    backbone, embedding = None, None

    if arch == "resnet18":
        backbone, embedding = resnet.__dict__[arch](
            zero_init_residual=True,
            num_channels=channels,
        )
    elif arch == "resnet18ID":
        backbone, embedding = resnet.__dict__[arch](
            zero_init_residual=False, num_channels=channels
        )
    elif arch == "lenet5":
        backbone = LeNet5()
        embedding = 128
    elif arch == "id":
        backbone = PassThrough()
        embedding = 28 * 28
    elif arch == "menet5":
        backbone = MeNet5(
            embedding_size, width_factor=backbone_width_factor, input_channels=channels
        )
        embedding = embedding_size
    elif arch == "mlp":
        backbone = MLPNet(embedding_size)
        embedding = embedding_size
    elif arch == "canonical":
        backbone = Canonical(embedding_size)
        embedding = embedding_size
    elif arch == "pixel":
        backbone = PixelEncoder()
        embedding = 512
    elif arch == "vit":
        backbone = VisionTransformer(
            img_size=28,
            patch_size=4,
            num_frames=18,
            embed_dim=kwargs['encoder_embed_dim'],
            depth=kwargs['encoder_depth'],
            num_heads=kwargs['encoder_num_heads'],
            mlp_ratio=kwargs['encoder_mlp_ratio'],
            qkv_bias=kwargs['encoder_qkv_bias'],
            qk_scale=kwargs['encoder_qk_scale'],
        )
        embedding = kwargs['encoder_embed_dim']
    else:
        raise NotImplementedError(f"backbone arch {arch} is unknown")

    if backbone_mlp is not None:
        backbone_mlp = Projector(backbone_mlp, embedding)
        embedding = backbone_mlp.output_dim
        backbone = nn.Sequential(backbone, backbone_mlp)

    return backbone, embedding


class Predictor(torch.nn.Module):
    def __init__(self, arch: str, num_features: int, action_dim: int = 2):
        super().__init__()
        layers = []
        f = (
            [num_features + action_dim]
            + (list(map(int, arch.split("-"))) if arch != "" else [])
            + [num_features]
        )
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x, action):
        t = torch.concat([x, action], dim=1)
        return self.model(t)

    def predict_sequence(self, h: torch.Tensor, actions: torch.Tensor):
        outputs = []
        for i in range(len(actions)):
            h = h + self(h, actions[i])
            outputs.append(h)
        return torch.stack(outputs, dim=0)


class IDPredictor(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def burn_in(self, *args, **kwargs):
        return None

    def predict_sequence(
        self, enc: torch.Tensor, h: torch.Tensor, actions: torch.Tensor
    ):
        return enc.unsqueeze(0).repeat(actions.shape[0], 1, 1)


class RNNPredictor(torch.nn.Module):
    def __init__(
        self, hidden_size: int = 512, num_layers: int = 1, action_dim: int = 2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = torch.nn.GRU(
            input_size=action_dim,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
        )

    def burn_in(self, *args, **kwargs):
        return None

    def predict_sequence(
        self, enc: torch.Tensor, h: torch.Tensor, actions: torch.Tensor
    ):
        # in this version, encoding is directly used as h, and the passed h is ignored.
        # since h is obtained from burn_in, it's actually None.
        h = enc
        return self.rnn(actions, h.unsqueeze(0).repeat(self.num_layers, 1, 1))[0]


class RNNPredictorBurnin(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        output_size: int = 512,
        num_layers: int = 1,
        action_dim: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.action_dim = action_dim

        self.rnn = torch.nn.GRU(
            input_size=action_dim + output_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
        )
        self.output_projector = nn.Linear(hidden_size, output_size)

    def burn_in(
        self,
        encs: torch.Tensor,
        actions: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ):
        """Runs a few iterations of RNN with the provided GT encodings to obtain h0"""
        if h is None:
            h = torch.zeros(self.num_layers, actions.shape[1], self.hidden_size).to(
                actions.device
            )

        for i in range(encs.shape[0]):
            rnn_input = torch.cat([encs[i], actions[i]], dim=1).unsqueeze(0)
            _, h = self.rnn(rnn_input, h)
        return h

    def predict_sequence(
        self, enc: torch.Tensor, actions: torch.Tensor, h: Optional[torch.Tensor] = None
    ):
        """Predicts the sequence given gt encoding for the current time step"""
        outputs = []
        if h is None:
            h = torch.zeros(self.num_layers, actions.shape[1], self.hidden_size).to(
                actions.device
            )
        for i in range(actions.shape[0]):
            rnn_input = torch.cat([enc, actions[i]], dim=1).unsqueeze(0)
            _, h = self.rnn(rnn_input, h)
            outputs.append(self.output_projector(h[-1]))
            enc = outputs[-1]  # autoregressive GRU
        outputs = torch.stack(outputs)
        return outputs


class ActionPredictor(torch.nn.Module):
    def __init__(self, embedding: int, action_dim: int = 3):
        super().__init__()
        self.model = nn.Linear(embedding * 2, action_dim)

    def forward(self, s, sn):
        t = torch.concat([s, sn], dim=1)
        return self.model(t)


class AttentivePooler(nn.Module):
    """ Attentive Pooler """
    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer)
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias)

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=False,
                    norm_layer=norm_layer)
                for i in range(depth-1)])

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        if self.blocks is not None:
            for blk in self.blocks:
                q = blk(q)
        return q


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=False,
        use_sdpa=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim*2), bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = (xattn @ v)

        q = q.transpose(1, 2).reshape(B, n, C)
        q = self.proj(q)
    
        return q


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q

class AttentivePooler(nn.Module):
    """ Attentive Pooler """
    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer)
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias)

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=False,
                    norm_layer=norm_layer)
                for i in range(depth-1)])

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        if self.blocks is not None:
            for blk in self.blocks:
                q = blk(q)
        return q


class AttentiveClassifier(nn.Module):
    """ Attentive Classifier """
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=1000,
        complete_block=True,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
        )
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        x = self.linear(x)
        return x

def build_predictor(arch: str, embedding: int, action_dim: int, rnn_layers: int, attentive_num_heads: int, attentive_depth: int):
    if arch == "conv":
        predictor = PixelPredictorConv(action_dim=action_dim)
    elif arch == "rnn":
        predictor = RNNPredictor(
            hidden_size=embedding,
            num_layers=rnn_layers,
            action_dim=action_dim,
        )
    elif arch == "rnn_burnin":
        predictor = RNNPredictorBurnin(
            hidden_size=embedding,
            output_size=embedding,
            num_layers=rnn_layers,
            action_dim=action_dim,
        )
    elif arch == "id":
        predictor = IDPredictor()
    elif arch == "attentive_classifier":
        predictor = AttentiveClassifier(
            embedding,
            num_classes=action_dim,
            num_heads=attentive_num_heads,
            depth=attentive_depth
        )
    else:
        predictor = Predictor(arch, embedding, action_dim=action_dim)

    return predictor


class Prober(torch.nn.Module):
    def __init__(self, embedding: int, arch: str, output_shape: List[int]):
        super().__init__()

        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        # f = [embedding, embedding, embedding]
        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.BatchNorm1d(f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1], bias=False))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        # return self.prober(e)
        return self.prober(e).view(e.shape[0], *self.output_shape)

###### VJEPA

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        use_sdpa=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob)
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        drop_path=0.
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x
    

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        patch_size=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        T, B, C, H, W = x.shape
        # This assumes B C T H W? But we have T B C H W
        x = self.proj(x.permute(1, 2, 0, 3, 4)).flatten(2).transpose(1, 2)
        # B (T P P) C
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=28,
        patch_size=4,
        num_frames=18,
        tubelet_size=2,
        in_chans=1,
        embed_dim=64,
        depth=12,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers

        self.input_size = img_size
        self.patch_size = patch_size

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1


        # Tokenize pixels with convolution
        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                embed_dim=embed_dim)
            self.num_patches = (
                (num_frames // tubelet_size)
                * (img_size // patch_size)
                * (img_size // patch_size)
            )
        else:
            self.patch_embed = PatchEmbed(
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim)
            self.num_patches = (
                (img_size // patch_size)
                * (img_size // patch_size)
            )

        # Position embedding
        self.uniform_power = uniform_power
        self.pos_embed = None
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim),
            requires_grad=False)

        # Attention Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.GELU,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # ------ initialize weights
        if self.pos_embed is not None:
            self._init_pos_embed(self.pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim,
                grid_size,
                grid_depth,
                cls_token=False,
                uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

    def forward(self, x: torch.Tensor, masks=None):
        """
        :param x: input image/video
        :param masks: indices of patch tokens to mask (remove)
        """

        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        # Tokenize input
        pos_embed = self.pos_embed
        if pos_embed is not None:
            pos_embed = self.interpolate_pos_encoding(x, pos_embed)
        x = self.patch_embed(x)
        if pos_embed is not None:
            x += pos_embed
        B, N, D = x.shape

        # Mask away unwanted tokens (if masks provided)
        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        # Fwd prop
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))

        if self.out_layers is not None:
            return outs

        if self.norm is not None:
            x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):

        _, N, dim = pos_embed.shape

        if self.is_video:

            # If pos_embed already corret size, just return
            T, _, _, H, W = x.shape
            if H == self.input_size and W == self.input_size and T == self.num_frames:
                return pos_embed

            # Convert depth, height, width of input to be measured in patches
            # instead of pixels/frames
            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size

            # Compute the initialized shape of the positional embedding measured
            # in patches
            N_t = self.num_frames // self.tubelet_size
            N_h = N_w = self.input_size // self.patch_size
            assert N_h * N_w * N_t == N, 'Positional embedding initialized incorrectly'

            # Compute scale factor for spatio-temporal interpolation
            scale_factor = (T/N_t, H/N_h, W/N_w)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode='trilinear')
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed

        else:

            # If pos_embed already corret size, just return
            _, _, H, W = x.shape
            if H == self.input_size and W == self.input_size:
                return pos_embed

            # Compute scale factor for spatial interpolation
            npatch = (H // self.patch_size) * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode='bicubic')
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed
        

class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=28,
        patch_size=4,
        num_frames=18,
        tubelet_size=2,
        embed_dim=64,
        predictor_embed_dim=32,
        depth=12,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        **kwargs
    ):
        super().__init__()
        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Mask tokens
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                for i in range(num_mask_tokens)
            ])

        # Determine positional embedding
        self.input_size = img_size
        self.patch_size = patch_size
        # --
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        if self.is_video:
            self.num_patches = num_patches = (
                (num_frames // tubelet_size)
                * (img_size // patch_size)
                * (img_size // patch_size)
            )
        else:
            self.num_patches = num_patches = (
                (img_size // patch_size)
                * (img_size // patch_size)
            )
        # Position embedding
        self.uniform_power = uniform_power
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim),
            requires_grad=False)

        # Attention Blocks
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.GELU,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer)
            for i in range(depth)])

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ------ initialize weights
        if self.predictor_pos_embed is not None:
            self._init_pos_embed(self.predictor_pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim,
                grid_size,
                grid_depth,
                cls_token=False,
                uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def diffusion(self, x, noise_beta=(0.5, 1.0), steps=1000):
        # Prepare diffusion noise schedule
        b1, b2 = noise_beta
        beta_scheduler = (b1 + i*(b2-b1)/steps for i in range(steps))
        alpha_scheduler = []
        _alpha = 1.0
        for _beta in beta_scheduler:
            _alpha *= 1.-_beta
            alpha_scheduler += [_alpha]

        # Sample diffusion time step
        T = torch.randint(0, steps, (len(x),))
        alpha = torch.tensor(alpha_scheduler, device=x.device)[T].unsqueeze(-1).unsqueeze(-1)

        # Normalize features and apply noise
        x = torch.nn.functional.layer_norm(x, (x.size(-1),))
        x = alpha**0.5 * x + (1.-alpha)**0.5 * torch.randn(x.shape, device=x.device)
        return x

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt, actions, mask_index=1):
        """
        :param ctxt: context tokens
        :param tgt: target tokens
        :param masks_ctxt: indices of context tokens in input
        :params masks_tgt: indices of target tokens in input
        """

        assert (masks_ctxt is not None) and (masks_tgt is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_ctxt, list):
            masks_ctxt = [masks_ctxt]

        if not isinstance(masks_tgt, list):
            masks_tgt = [masks_tgt]

        # Batch Size
        B = len(ctxt) // len(masks_ctxt)

        # Map context tokens to pedictor dimensions
        x = self.predictor_embed(ctxt)
        _, N_ctxt, D = x.shape

        # Add positional embedding to ctxt tokens
        ctxt_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(ctxt_pos_embed, masks_ctxt)

        # Map target tokens to predictor dimensions & add noise (fwd diffusion)
        if self.mask_tokens is None:
            pred_tokens = self.predictor_embed(tgt)
            pred_tokens = self.diffusion(pred_tokens)
        else:
            mask_index = mask_index % self.num_mask_tokens
            pred_tokens = self.mask_tokens[mask_index]
            pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
            pred_tokens = apply_masks(pred_tokens, masks_tgt)

        # Add positional embedding to target tokens
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks_tgt)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_ctxt))
        pred_tokens += pos_embs

        # Concatenate context & target tokens
        x = x.repeat(len(masks_tgt), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # Fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # Return output corresponding to target tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x
if __name__ == "__main__":
    model = LeNet5()
    test_in = torch.rand(1, 1, 28, 28)
    print(model(test_in).shape)
