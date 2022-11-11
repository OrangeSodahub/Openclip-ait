""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Union, Optional

from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.compiler.ops.common.epilogue import FuncEnum

from .utils import to_2tuple


USE_CUDA = detect_target().name() == "cuda"


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, x):
        x1 = x * 1.702
        x1 = ops.sigmoid(x1)
        x = x * x1
        return x


class CLIPAttention(nn.Module):
    """
    adapted from torch.nn.attention, no `flash_attention`
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        causal=False,
        mask_seq=0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.causal = causal
        self.mask_seq = mask_seq

        if USE_CUDA:
            self.qkv_weight = nn.Parameter(
                shape=[dim * 3, dim],
                dtype="float16"
            )
            if qkv_bias:
                self.qkv_bias = nn.Parameter(
                    shape=[dim * 3],
                    dtype="float16"
                )
        else:
            raise RuntimeError("no CUDA!")

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, specialization="add")
        self.proj_drop = nn.Dropout(proj_drop)
    
    def get_shape(self, x):
        shape = [it.value() for it in x._attrs["shape"]]
        return shape   

    def qkv_proj(self, x: Tensor):
        """
        Performs the in-projection step of teh attention operation, using packed weights.
        Output is a triple containing projection tensors for query, key and value.
        :param x: q == k == v == x of shape `(batch_size, context_length, d_model)`
        """
        # sefl-attention
        batch, seqlen, hidden = self.get_shape(x)
        # 'b s hd -> (b s) hd
        x = ops.reshape()(x, [-1, hidden])
        # compute in projection
        x = ops.gemm_rcr_bias()(
                x,
                self.qkv_weight.tensor(),
                self.qkv_bias.tensor()
            )
        x = ops.reshape()(x, [batch, seqlen, hidden * 3])
        return ops.chunk()(x, chunks=3, dim=-1)

    def attention(self, q: Tensor, k: Tensor, v: Tensor):
        """
        Computes scaled dot product attention on query, key and value tensors. No weights.
        :param q: qurey tensor of shape `(batch_size * num_heads, context_length, head_dim)`
        :param k: key tensor of shape `(batch_size * num_heads, context_length, head_dim)`
        :param v: value tensor of shape `(batch_size * num_heads, context_length, head_dim)`
        """
        scale = Tensor(
            shape=[], dtype="float16", name="scale", value=self.scale
        )
        q = ops.elementwise(FuncEnum.MUL)(q, scale)
        # (B*NH, C, E) x (B*NH, E, C) -> (B*NH, C, C)
        attn = ops.bmm_rrr()(q, ops.permute021()(k))
        attn = ops.softmax()(attn, -1)
        # (B*NH, C, C) x (B*NH, C, E) -> (B*NH, C, E)
        output = ops.bmm_rrr()(attn, v)
        return output

    # TODO: attn_mask
    def forward(self, x: Tensor, residual: Optional[Tensor] = None):
        """forward pass for calling mha module"""
        batch, seqlen, hidden = self.get_shape(x)
        # input: `(batch, seqlen, hidden)`
        # output: `(batch, seqlen, hidden)`
        q, k, v = self.qkv_proj(x)
        # `b s (h d) -> s b (h d)`
        q = ops.permute102()(q)
        k = ops.permute102()(k)
        v = ops.permute102()(v)
        # `s b (h d) -> s (b h) d`
        q = ops.reshape()(q, [seqlen, batch * self.num_heads, hidden // self.num_heads])
        k = ops.reshape()(k, [seqlen, batch * self.num_heads, hidden // self.num_heads])
        v = ops.reshape()(v, [seqlen, batch * self.num_heads, hidden // self.num_heads])
        # `s (b h) d -> (b h) s d`
        q = ops.permute102()(q)
        k = ops.permute102()(k)
        v = ops.permute102()(v)
        # input: `(batch*num_heads, seqlen, head_dim)`
        # output: `(batch*num_heads, seqlen, head_dim)`
        attn_output = self.attention(q, k, v)
        # `(b h) s d -> s b (h d)`
        attn_output = ops.reshape()(
            ops.permute102()(attn_output),
            [seqlen, batch, hidden]
        )
        # `s b (h d) -> b s (h d)`
        attn_output = ops.permute102()(attn_output)
        attn_output = self.proj(attn_output, residual)
        attn_output = self.proj_drop(attn_output)
        return attn_output


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
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.attn = CLIPAttention(
            dim=d_model,
            num_heads=n_head,
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
        """
        # TODO: attn_mask
        residual = x
        x = self.ln_attn(self.attn(self.ln_1(x), residual))
        x = x + self.mlp(self.ln_2(x))

        return x


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

        # Expand channel from 3 to 4, Also Expand conv1_weight to [width, 4, patch_size, patch_size])
        # In AIT, use Conv2dBiasFewChannels when channels < 4, set bias to zeros
        self.conv1 = nn.Conv2dBiasFewChannels(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.class_embedding = nn.Parameter(
            shape=[width],
            dtype="float16"
        )
        self.positional_embedding = nn.Parameter(
            shape=[self.grid_size[0] * self.grid_size[1] + 1, width],
            dtype="float16"
        )
        self.ln_pre = nn.LayerNorm(width)

        # TODO: flash_attn
        seq_len = self.grid_size[0] * self.grid_size[1] + 1

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

        # expand shape[0] to batch_size
        class_embedding = ops.unsqueeze(0)(
            ops.unsqueeze(0)(self.class_embedding.tensor())
        )
        class_embedding = ops.concatenate()(
            [class_embedding for _ in range(x.shape()[0].value())],
            dim=0
        )
        x = ops.concatenate()([class_embedding, x], dim=1)
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
        self.transformer = Transformer(
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            act_layer=act_layer,
        )

        self.vocab_size = text_cfg.vocab_size
        self.token_embedding = nn.Embedding(shape=[text_cfg.vocab_size, text_cfg.width], dtype="float16")
        self.positional_embedding = nn.Parameter(shape=[self.context_length, text_cfg.width], dtype="float16") # no initialization
        self.ln_final = nn.LayerNorm(text_cfg.width, dtype="float16")

        self.text_projection = nn.Parameter(shape=[text_cfg.width, embed_dim], dtype="float16")
        self.logit_scale = nn.Parameter(shape=[], dtype="float16")

    def encode_text(self, text: Tensor):
        """
        :param text: of shape `(batch_size, context_length)`
        In Pytorch, when `batch_first` is False the nn.MultiheadAttention gets input of shape :math:`(L, N, E_q)` -> `(context_length, batch_size, d_model)`
        In AIT, nn.MultiheadAttention gets input of shape `(batch_size, context_length, d_model)`
        """
        input_shape = ops.size()(text)
        
        text = ops.reshape()(text, [-1]) # [batch_size * context_length]
        x = ops.batch_gather()(self.token_embedding.tensor(), text)  # [batch_size, n_ctx, d_model]
        x = ops.reshape()(x, [input_shape[0], input_shape[1], -1])

        x = x + self.positional_embedding.tensor()
        x = self.transformer(x)
        x = self.ln_final(x)

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
        input_shape = ops.size()(text)
        # [B * S]
        text = ops.reshape()(text, [-1])
        x = ops.batch_gather()(self.token_embedding.tensor(), text)  # [batch_size, n_ctx, d_model]
        x = ops.reshape()(x, [input_shape[0], input_shape[1], -1])

        x = x + self.positional_embedding.tensor()
        x = self.transformer(x)
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
