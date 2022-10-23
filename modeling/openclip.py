""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from collections import OrderedDict
from dataclasses import dataclass
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

    def attention(self, x: Tensor, attn_mask: Optional[Tensor] = None):
        return self.attn(x, x, x)[0]
        # FIXME torchscript issues need resolving for custom attention option to work
        # if self.use_torch_attn:
        #     return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # else:
        #     return self.attn(x, attn_mask=attn_mask)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # TODO: verify
        # now the shape of hidden_states: seqlen, batch_size, d_model
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states += self.ln_attn(hidden_states)
        hidden_states = self.attn(hidden_states, residual)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states) # TODO

        return hidden_states


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


class VisualTransformer(nn.Module):
    def __init__(
            self,
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

        self.transformer = Transformer(
            width=width,
            layers=layers,
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

        # Concat cls token: shape = [*, grid ** 2 + 1, width]
        x = ops.concatenate()([self.class_embedding.tensor(), x], dim=1)
        # Concat pos token: shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.tensor()
        x = self.ln_pre(x)

        x = ops.permute()(x, (1, 0, 2))  # NLD -> LND
        x = self.transformer(x)
        x = ops.permute()(x, (1, 0, 2))  # LND -> NLD

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

    def forward(self, text = None, *args, **kwargs):
        text = text or args
        return self.encode_text(text)


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
