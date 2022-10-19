import click
import logging
from regex import R
import torch
import numpy as np

from aitemplate.testing import detect_target
from aitemplate.compiler import compile_model
from aitemplate.frontend import Tensor
from diffusers import StableDiffusionPipeline
from modeling.clip import CLIPTextTransformer as ait_CLIPTextTransformer
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
        name = key.replace("transformer", "encoder")
        name = name.replace("resblocks", "layers")
        name = name.replace("positional_embedding", "embeddings.position_embedding.weight")
        name = name.replace("token_embedding", "embeddings.token_embedding")
        name = name.replace("ln_1", "layer_norm1")
        name = name.replace("ln_2", "layer_norm2")
        name = name.replace("ln_final", "final_layer_norm")
        name = name.replace("attn", "self_attn")
        name = name.replace("c_fc", "fc1")
        name = name.replace("c_proj", "fc2")
        ait_name = name.replace(".", "_")
        if name.startswith("visual") or name == 'logit_scale' or name == 'text_projection':
            continue
        if name.endswith("out_proj.weight"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("out_proj.bias"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("in_proj_weight"):
            ait_name = ait_name.replace("in_proj", "qkv")
        elif name.endswith("in_proj_bias"):
            ait_name = ait_name.replace("in_proj", "qkv")

        if ('fina_layer_norm' in ait_name
            or ait_name == 'embeddings_token_embedding_weight'
            or ait_name == 'embeddings_position_embedding_weight'
        ):
            arr.data = arr.data.half()
        print(ait_name)
        params_ait[ait_name] = arr

        if USE_CUDA:
            for i in range(depth):
                prefix = "encoder_layers_%d_self_attn_cu_length" % (i)
                cu_len = np.cumsum([0] + [seqlen] * batch_size).astype("int32")
                params_ait[prefix] = torch.from_numpy(cu_len).cuda()

    return params_ait


def compile_clip(
    batch_size=1,
    seqlen=64,
    dim=768,
    num_heads=12,
    hidden_size=768,
    vocab_size=49408,
    max_position_embeddings=77,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
):
    mask_seq = 0
    causal = True
    depth = 12

    ait_mod = ait_CLIPTextTransformer(
        num_hidden_layers=depth,
        hidden_size=dim,
        num_attention_heads=num_heads,
        batch_size=batch_size,
        seq_len=seqlen,
        causal=causal,
        mask_seq=mask_seq,
    )
    ait_mod.name_parameter_tensor()

    # load pytorch model
    openclip_mod = OpenCLIPModel(name='ViT-B-32::laion400m_e31', device='cuda')
    # textmodel
    pt_mod = openclip_mod._model
    pt_mod = pt_mod.eval()
    params_ait = map_clip_params(pt_mod, batch_size, seqlen, depth)

    input_ids_ait = Tensor(
        [batch_size, seqlen], name="input0", dtype="int64", is_input=True
    )
    position_ids_ait = Tensor(
        [batch_size, seqlen], name="input1", dtype="int64", is_input=True
    )
    Y = ait_mod(input_ids=input_ids_ait, position_ids=position_ids_ait)
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


    width = 96 if img2img else 64

    # CLIP
    compile_clip(batch_size=batch_size, use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm)


if __name__=="__main__":
    compile()