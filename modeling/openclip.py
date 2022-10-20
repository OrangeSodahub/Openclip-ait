""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from collections import OrderedDict
from dataclasses import dataclass
import logging
import math
from typing import Tuple, Union, Callable, Optional

import numpy as np
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from .timm_model import TimmModel
from .utils import freeze_batch_norm_2d, to_2tuple


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, x):
        x1 = x * 1.702
        x1 = ops.sigmoid(x1)
        x = x * x1
        return x


# CLIPEncoderLayer
class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            act_layer,
            attention_dropout = 0.0,
            mlp_ratio: float = 4.0,
            batch_size = 1,
            seq_len = 16,
            causal = False,
            mask_seq = 0,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        # FIXME torchscript issues need to be resolved for custom attention
        # if scale_cosine_attn or scale_heads:
        #     self.attn = Attention(
        #        d_model, n_head,
        #        scaled_cosine=scale_cosine_attn,
        #        scale_heads=scale_heads,
        #     )
        self.attn = nn.MultiheadAttention(
            dim=d_model,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=n_head,
            attn_drop=attention_dropout,
            causal=causal,
            mask_seq=mask_seq
        )
        self.ln_attn = nn.LayerNorm(d_model) if scale_attn else nn.Identity()

        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        # CLIPMLP
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ('ln', nn.LayerNorm(mlp_width) if scale_fc else nn.Identity()),
            ("gelu", act_layer),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))

    def attention(self, x: Tensor, attn_mask: Optional[Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # FIXME torchscript issues need resolving for custom attention option to work
        # if self.use_torch_attn:
        #     return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # else:
        #     return self.attn(x, attn_mask=attn_mask)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None):
        x = x + self.ln_attn(self.attention(self.ln_1(x), attn_mask=attn_mask))
        x = x + self.mlp(self.ln_2(x))
        return x


# CLIPEncoder
class Transformer(nn.Module):
    def __init__(self,
        width: int,                                 # hidden_size
        layers: int,                                # num_hidden_layers
        heads: int,                                 # num_attention_heads
        act_layer: QuickGELUActivation,
        mlp_ratio: float = 4.0,
        batch_size: int = 1,
        seq_len: int = 64,
        causal = False,
        mask_seq: int = 0,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=width,
                n_head=heads,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                batch_size=batch_size,
                seq_len=seq_len,
                causal=causal,
                mask_seq=mask_seq,
            )
            for _ in range(layers)
        ])

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None):
        for r in self.resblocks:
            # TODO: torch: checkpoint
            # if self.grad_checkpointing and not torch.jit.is_scripting():
            #     x = checkpoint(r, x, attn_mask)
            # else:
                x = r(x, attn_mask=attn_mask)
        return x


@dataclass
class CLIPTextCfg:
    context_length: int = 77        # CLIPTextEmbeddings.max_position_embeddings
    vocab_size: int = 49408         # CLIPTextEmbeddings.vocab_size
    width: int = 512                # hidden_size
    heads: int = 8                  # num_attention_heads
    layers: int = 12                # num_hidden_layers


class CLIPTextTransformer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            text_cfg: CLIPTextCfg,
            batch_size: int = 1,
            seq_len: int = 64,
            causal = False,
            mask_seq = 0,
    ):
        super().__init__()
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)

        self.context_length = text_cfg.context_length

        # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
        # memory efficient in recent PyTorch releases (>= 1.10).
        # NOTE: timm models always use native GELU regardless of quick_gelu flag.
        act_layer = QuickGELUActivation()

        # self.encoder = CLIPEncoder(...)
        self.transformer = Transformer(
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            act_layer=act_layer,
            batch_size=batch_size,
            seq_len=seq_len,
            causal=causal,
            mask_seq=mask_seq,
        )

        self.vocab_size = text_cfg.vocab_size
        self.token_embedding = nn.Embedding(shape=[text_cfg.vocab_size, text_cfg.width], dtype="float16")
        self.positional_embedding = nn.Parameter(shape=[self.context_length, text_cfg.width], dtype="float16") # no initialization
        self.ln_final = nn.LayerNorm(text_cfg.width)

        self.text_projection = nn.Parameter(shape=[text_cfg.width, embed_dim], dtype="float16")
        self.logit_scale = nn.Parameter(shape=[], dtype="float16")

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        return x

    def forward(self, text):
        text_features = self.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        return text_features, self.logit_scale.exp()