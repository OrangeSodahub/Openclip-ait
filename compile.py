import click
import logging
from regex import R
import torch
import numpy as np

from aitemplate.testing import detect_target
from aitemplate.compiler import compile_model
from aitemplate.frontend import Tensor
from modeling.openclip import CLIP as ait_CLIP
from modeling.openclip_model import OpenCLIPModel


USE_CUDA = detect_target().name() == "cuda"
pipe = None


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("AIT output_{} shape: {}".format(i, y_shape))


def map_clip_params(pt_mod, batch_size, seqlen, depth):

    params_ait = {}
    pt_params = {}

    pt_params = dict(pt_mod.named_parameters())
    for key, arr in pt_params.items():
        name = key
        ait_name = name.replace(".", "_")
        if name.endswith("out_proj.weight"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("out_proj.bias"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("in_proj_weight"):
            ait_name = ait_name.replace("in_proj", "qkv")
        elif name.endswith("in_proj_bias"):
            ait_name = ait_name.replace("in_proj", "qkv")

        if arr.dtype == torch.float32:
            arr.data = arr.data.half()
        print(f"name:{ait_name}, shape:{arr.shape}")
        params_ait[ait_name] = arr

        # TODO: prefix changed
        if USE_CUDA:
            for i in range(depth):
                prefix = "transformer_resblocks_%d_attn_cu_length" % (i)
                cu_len = np.cumsum([0] + [seqlen] * batch_size).astype("int32")
                params_ait[prefix] = torch.from_numpy(cu_len).cuda()

    return params_ait


# ATTENTION: the cfgs of model
def compile_clip(
    embed_dim,
    text_cfg,
    vision_cfg,
    batch_size=1,
    seqlen=64,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
):
    mask_seq = 0
    causal = True
    depth = 12

    ait_mod = ait_CLIP(
        embed_dim = embed_dim,
        text_cfg = text_cfg,
        vision_cfg = vision_cfg,
        batch_size = batch_size,
        seq_len = seqlen,
        causal = causal,
        mask_seq = mask_seq,
    )
    ait_mod.name_parameter_tensor()

    # load pytorch model
    openclip_mod = OpenCLIPModel(name='ViT-L-14::laion400m_e31', device='cuda')
    pt_mod = openclip_mod._model
    pt_mod = pt_mod.eval()
    params_ait = map_clip_params(pt_mod, batch_size, seqlen, depth)
    print(f"num of params: {len(params_ait)}")

    # text input
    input_text_ait = Tensor(
        [batch_size, text_cfg['context_length']], name="input0", dtype="int64", is_input=True
    )
    # image input
    # input tensor: N, H, W, C_in (ait)
    #               N, C_in, H, W (torch)
    input_image_ait = Tensor(
        [batch_size, vision_cfg['image_size'], vision_cfg['image_size'], 3], name="input1", dtype="float16", is_input=True
    )
    Y = ait_mod(text=input_text_ait, image=input_image_ait)
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(Y, target, "./tmp", "CLIPTextModel", constants=params_ait)



@click.command()
@click.option("--batch-size", default=1, help="batch size")
@click.option("--img2img", default=False, help="compile img2img models")
@click.option("--use-fp16-acc", default=True, help="use fp16 accumulation")
@click.option("--convert-conv-to-gemm", default=True, help="convert 1x1 conv to gemm")
def compile(batch_size, img2img=False, use_fp16_acc=True, convert_conv_to_gemm=True):
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    # cfgs for model
    text_cfg = {
        'layers': 12,
        'context_length': 77,           # max_position_embeddings
        'vocab_size': 49408,            # vocab_size
        'width': 768,                   # hidden_size
        'heads': 12,                    # num_heads
    },
    vision_cfg = {
        'layers': 24,
        'width': 1024,
        'head_width': 64,
        'mlp_ratio': 4.,
        'patch_size': 14,
        'image_size': 224,
    },

    # CLIP
    compile_clip(
        embed_dim=768,
        text_cfg=text_cfg[0],
        vision_cfg=vision_cfg[0],
        batch_size=batch_size,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm
    )


if __name__=="__main__":
    compile()