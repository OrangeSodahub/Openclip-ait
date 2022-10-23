import click
import logging
from regex import R
import torch
import numpy as np

from aitemplate.testing import detect_target
from aitemplate.compiler import compile_model
from aitemplate.frontend import Tensor
from modeling.openclip import CLIPVisionTransformer as ait_CLIP
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


def map_clip_params(pt_mod, width, patch_size, depth, seqlen, batch_size):

    params_ait = {}
    pt_params = {}

    pt_params = dict(pt_mod.named_parameters())
    for key, arr in pt_params.items():
        name = key
        ait_name = name.replace(".", "_")
        if not name.startswith("visual"):
            continue
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

        if name.startswith("visual.conv1"):
            conv_w = torch.zeros((width, 4, patch_size, patch_size), dtype=torch.float16)
            conv_w[:, :3, :, :] = arr
            arr = conv_w.permute((0, 2, 3, 1))                                               # [N, C, H, W] -> [N, H, W, C]
            params_ait["visual_conv1_weight"] = arr
            params_ait["visual_conv1_bias"] = torch.zeros((width), dtype=torch.float16)      # Set bias to zero
            continue

        print(f"name:{ait_name}, shape:{arr.shape}")
        params_ait[ait_name] = arr

        if USE_CUDA:
            for i in range(depth):
                prefix = "visual_transformer_resblocks_%d_attn_cu_length" % (i)
                cu_len = np.cumsum([0] + [seqlen] * batch_size).astype("int32")
                params_ait[prefix] = torch.from_numpy(cu_len).cuda()

    return params_ait


# ATTENTION: the cfgs of model
def compile_clip(
    embed_dim,
    vision_cfg,
    batch_size=1,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
):
    ait_mod = ait_CLIP(
        embed_dim = embed_dim,
        vision_cfg = vision_cfg,
    )
    ait_mod.name_parameter_tensor()

    # TODO: + (1 if class_token else 0)
    seqlen = (vision_cfg["image_size"] // vision_cfg["patch_size"]) ** 2

    # load pytorch model
    openclip_mod = OpenCLIPModel(name='ViT-L-14::laion400m_e31', device='cuda')
    pt_mod = openclip_mod._model
    pt_mod = pt_mod.eval()
    params_ait = map_clip_params(
        pt_mod=pt_mod,
        width=vision_cfg['width'],
        patch_size=vision_cfg['patch_size'],
        depth=vision_cfg['layers'],
        seqlen=seqlen,
        batch_size=batch_size,
    )
    print(f"num of params: {len(params_ait)}")

    # image input
    # input tensor: N, H, W, C_in (ait)
    #               N, C_in, H, W (torch)
    input_image_ait = Tensor(
        [batch_size, vision_cfg['image_size'], vision_cfg['image_size'], 3], name="input1", dtype="float16", is_input=True
    )
    Y = ait_mod(image=input_image_ait)
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    # TODO: Error: Constant tensor_0 was not set! Set the value with set_constant.
    compile_model(Y, target, "./tmp", "CLIPTextModel", constants=params_ait)



@click.command()
@click.option("--batch-size", default=1, help="batch size")
@click.option("--use-fp16-acc", default=True, help="use fp16 accumulation")
@click.option("--convert-conv-to-gemm", default=True, help="convert 1x1 conv to gemm")
def compile(batch_size, use_fp16_acc=True, convert_conv_to_gemm=True):
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    # cfgs for model
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
        vision_cfg=vision_cfg[0],
        batch_size=batch_size,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm
    )


if __name__=="__main__":
    compile()