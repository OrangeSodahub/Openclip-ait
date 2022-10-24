""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from collections import OrderedDict
from dataclasses import dataclass
from turtle import forward
from typing import Tuple, Union, Optional

from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target

from .utils import to_2tuple


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
    """
    :param d_model: hidden_size, comes from text_cfg.width
    :param n_head: num_attention_heads, comes from text_cfg.heads
    :param act_layer: use QuickGELUActivation
    :mlp: c_fc(fc1) of shape (in_features, hidden_features)
          activation_fn
          c_proj(fc2) of shape (hidden_features, out_features)
    """

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
            mask_seq=mask_seq,
            has_residual=False,
            qkv_bias=True,
        )
        self.ln_attn = nn.LayerNorm(d_model, dtype="float16") if scale_attn else nn.Identity()

        self.ln_1 = nn.LayerNorm(d_model, dtype="float16")
        self.ln_2 = nn.LayerNorm(d_model, dtype="float16")
        mlp_width = int(d_model * mlp_ratio)
        # CLIPMLP
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width, dtype="float16")),
            ('ln', nn.LayerNorm(mlp_width, dtype="float16") if scale_fc else nn.Identity()),
            ("gelu", act_layer),
            ("c_proj", nn.Linear(mlp_width, d_model, dtype="float16"))
        ]))

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
    ):
        """
        :param x: of shape `(batch_size, context_length, d_model)`
        :param attn_mask:
        attn:
            input: of shape `(batch_size, context_length, d_model)`
            output: of shape `(batch_size, context_length, d_model)`
        
        In pytorch, nn.MultiheadAttention does not operate the `add`
        where the forward code like:
            ```
            x = self.attention(self.ln_1(x))
            x = x + self.ln_attn(x, attn_mask=attn_mask))
            x = x + self.mlp(self.ln_2(x))
            ```
        In AIT, nn.MultiheadAttention accept `*args` which support two arguments,
        if the seconde arg `residual` specified, then it will do `add` operation.
        MLP layer is not the same as the Attn layer. In AIT, nn.Linear accept `specialization="add"` in `fc2`.
        """
        # TODO: attn_mask
        x = self.ln_1(x)
        x = x + self.ln_attn(self.attn(x))

        x = self.ln_2(x)
        x = x + self.mlp(x)

        return x


# CLIPEncoder
class Transformer(nn.Module):
    """
    :param width: hidden_size, comes from text_cfg.width
    :param layers: num_hidden_layers, comes from text_cfg.layers
    :param heads: num_attention_heads, comes from text_cfg.heads
    """
    
    def __init__(self,
        width: int,
        layers: int,
        heads: int,
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
        """
        :param x: shape: of shape (context_length, batch_size, d_model/width)
        :param attn_mask: of shape (context_length, batch_size)
               Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`
        """
        for r in self.resblocks:
            # TODO: grad_checkpointing
            x = r(x, attn_mask=attn_mask)
        return x


class VisualTransformer(nn.Module):
    def __init__(
            self,
            batch_size: int,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            output_dim: int,
            act_layer: QuickGELUActivation,
    ):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.output_dim = output_dim
        # Expand channel from 3 to 4
        # Also Expand conv1_weight to [width, 4, patch_size, patch_size])
        self.conv1 = nn.Conv2dBiasFewChannels(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size
        ) # bias = Flase => Conv2d == Conv2dBias == Conv2dBiasFewChannels

        # TODO: support batch_size > 1
        self.class_embedding = nn.Parameter(
            shape=[1, 1, width],
            dtype="float16"
        )
        self.positional_embedding = nn.Parameter(
            shape=[self.grid_size[0] * self.grid_size[1] + 1, width],
            dtype="float16"
        )
        self.ln_pre = nn.LayerNorm(width)
        seq_len = self.grid_size[0] * self.grid_size[1] + 1

        self.transformer = Transformer(
            width=width,
            layers=layers,
            batch_size=batch_size,
            # TODO: verify
            seq_len=seq_len,
            heads=heads,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer
        )

        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(shape=[width, output_dim], dtype="float16")

    def forward(self, x: Tensor):
        # Patch embedding: shape = [*, width, grid, grid] (pt) / [*, grid, grid, width]
        x = self.conv1(x)
        # Flatten to tokens: shape = [*, width, grid ** 2] (pt) / [*, grid ** 2, width]
        x = ops.reshape()(x, [x.shape()[0].value(), -1, x.shape()[3].value()]) 

        # TODO: Error: Constant tensor_0 was not set! Set the value with set_constant.
        # cls_token_mask
        # zeros = [
        #     [[
        #     0 for _ in range(x.shape()[-1].value())
        # ]] for _ in range(x.shape()[0].value())
        # ]
        # zeros = Tensor([x.shape()[0].value(), 1, x.shape()[-1].value()], dtype="float16", value=zeros)

        # TODO: tensors expected to have the same dimensions except concat_dim!
        # expand shape[0] to batch_size
        class_embedding = ops.expand()(
            self.class_embedding.tensor(), [x.shape()[0].value(), -1, -1]
        )
        # Concat cls token: shape = [*, grid ** 2 + 1, width]
        x = ops.concatenate()([class_embedding, x], dim=1)
        # Concat pos token: shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.tensor()
        x = self.ln_pre(x)

        x = self.transformer(x)

        x = ops.dynamic_slice()(
            x=x,
            start_indices=[0, 0, 0],
            end_indices=[
                x.shape()[0].value(),
                1,
                x.shape()[2].value(),
            ]
        )

        if self.proj is not None:
            x = ops.bmm_rrr()(x, self.proj.tensor())

        return x


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')


@dataclass
class CLIPTextCfg:
    context_length: int = 77        # CLIPTextEmbeddings.max_position_embeddings
    vocab_size: int = 49408         # CLIPTextEmbeddings.vocab_size
    width: int = 512                # hidden_size
    heads: int = 8                  # num_attention_heads
    layers: int = 12                # num_hidden_layers


class CLIPVisionTransformer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            batch_size: int,
            vision_cfg: CLIPVisionCfg,
    ):
        super().__init__()
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)

        # Always using QuickGELUActivation()
        act_layer = QuickGELUActivation()

        # Vision Transformer
        # layers name: visual....
        vision_heads = vision_cfg.width // vision_cfg.head_width
        self.visual = VisualTransformer(
            batch_size=batch_size,
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            output_dim=embed_dim,
            act_layer=act_layer,
        )

    def encode_image(self, image):
        return self.visual(image)

    def forward(self, image = None, *args, **kwargs):
        image = image or args
        return self.encode_image(image)


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

        # Always using QuickGELUActivation()
        act_layer = QuickGELUActivation()

        # General Transformer
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
        self.ln_final = nn.LayerNorm(text_cfg.width, dtype="float16")

        self.text_projection = nn.Parameter(shape=[text_cfg.width, embed_dim], dtype="float16")
        self.logit_scale = nn.Parameter(shape=[], dtype="float16")

    def encode_text(self, text: Tensor):
        """
        :param text: of shape (batch_size, context_length)
        In Pytorch, when `batch_first` is False the nn.MultiheadAttention gets input of shape :math:`(L, N, E_q)` -> `(context_length, batch_size, d_model)`
        In AIT, nn.MultiheadAttention gets input of shape `(batch_size, context_length, d_model)`
        See aitemplate.fronted.nn.attention.MultiheadAttention.forward
        """
        # CLIPTextEncodings.forward
        input_shape = ops.size()(text)
        
        text = ops.reshape()(text, [-1]) # [batch_size * context_length]
        x = ops.batch_gather()(self.token_embedding.tensor(), text)  # [batch_size, n_ctx, d_model]
        x = ops.reshape()(x, [input_shape[0], input_shape[1], -1])

        x = x + self.positional_embedding.tensor()
        x = self.transformer(x)
        # x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # a = Tensor(shape=[1, x.shape()[0].value()], value=[i for i in range(x.shape()[0].value())])

        # TODO: better way to index
        # index = ops.argmax(dim=-1)(x)
        # x = ops.bmm_rrr()(x[a, ops.argmax(dim=-1)(x)], self.text_projection)
        
        return x

    def forward(self, text = None, *args, **kwargs):
        text = text or args
        return self.encode_text(text)


# TODO: Not available now
class CLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            text_cfg: CLIPTextCfg,
            vision_cfg: CLIPVisionCfg,
            batch_size: int = 1,
            seq_len: int = 64,
            causal = False,
            mask_seq = 0,
    ):
        super().__init__()
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)

        self.context_length = text_cfg.context_length

        # Always using QuickGELUActivation()
        act_layer = QuickGELUActivation()

        # Vision Transformer
        vision_heads = vision_cfg.width // vision_cfg.head_width
        self.visual = VisualTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            output_dim=embed_dim,
            act_layer=act_layer,
        )

        # General Transformer
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
        self.ln_final = nn.LayerNorm(text_cfg.width, dtype="float16")

        self.text_projection = nn.Parameter(shape=[text_cfg.width, embed_dim], dtype="float16")
        self.logit_scale = nn.Parameter(shape=[], dtype="float16")

    def encode_text(self, text: Tensor):
        # CLIPTextEncodings.forward
        input_shape = ops.size()(text)
        # [B * S]
        text = ops.reshape()(text, [-1])
        x = ops.batch_gather()(self.token_embedding.tensor(), text)  # [batch_size, n_ctx, d_model]
        x = ops.reshape()(x, [input_shape[0], input_shape[1], -1])

        x = x + self.positional_embedding.tensor()
        x = ops.permute()(x, (1, 0, 2)) # NLD -> LND
        x = self.transformer(x)
        x = ops.permute()(x, (1, 0, 2))  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        a = Tensor(shape=[1, x.shape()[0].value()], value=[i for i in range(x.shape()[0].value())])

        # TODO: better way to index
        # index = ops.argmax(dim=-1)(x)
        # x = ops.bmm_rrr()(x[a, ops.argmax(dim=-1)(x)], self.text_projection)
        
        return x

    def encode_image(self, image):
        return self.visual(image)

    def forward(self, text = None, image = None, *args, **kwargs):
        # TODO: verify
        if text is not None and image is None:
            return self.encode_text(text)
        elif image is not None and text is None:
            return self.encode_image(image)
        
        text_features = self.encode_text(text)
        image_features = self.encode_image(image)

        return text_features, image_features
