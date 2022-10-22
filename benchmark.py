#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import torch
import logging
import click
import numpy as np
from PIL import Image

from open_clip.transform import image_transform
from aitemplate.compiler import Model
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from modeling.openclip_model import OpenCLIPModel

from torch import autocast
# from transformers import CLIPTokenizer
from clip_server.model.tokenization import Tokenizer
from open_clip import tokenizer

USE_CUDA = detect_target().name() == "cuda"

access_token = True
pipe = None


def get_int_shape(x):
    shape = [it.value() for it in x._attrs["shape"]]
    return shape


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("AIT output_{} shape: {}".format(i, y_shape))


def benchmark_clip(
    batch_size=1,
    seqlen=64,
    dim=512,
    num_heads=8,
    hidden_size=512,
    vocab_size=49408,
    max_position_embeddings=77,
    benchmark_pt=False,
    verify=False,
):
    mask_seq = 0
    version = "ViT-B-32::laion400m_e31"

    exe_module = Model("/home/zonlin/Jina/openclip-ait/tmp2/CLIPTextModel/test.so")
    if exe_module is None:
        print("Error!! Cannot find compiled module for CLIPTextModel.")
        exit(-1)

    # run PT clip
    openclip_mod = OpenCLIPModel(name='ViT-L-14::laion400m_e31', device='cuda')
    pt_mod = openclip_mod._model
    pt_mod = pt_mod.eval()

    # TODO: wrong inputs
    text = tokenizer.tokenize(["a diagram"]).cuda()
    preprocess = image_transform(224, is_train=False)
    # for test
    input_ait = torch.ones((1, 2, 77), dtype=torch.int64).long().cuda()
    input_pt = torch.ones((2, 77), dtype=torch.int64).long().cuda()

    # attention_mask = torch.ones((batch_size, seqlen))
    # attention_mask[-1, -mask_seq:] = 0
    # attention_mask = None

    # position_ids = torch.arange(seqlen).expand((batch_size, -1)).cuda()
    # pt_ys = pt_mod(input_ids, attention_mask, position_ids)
    # print("pt output:", pt_ys[0].shape)

    # PT benchmark
    if benchmark_pt:
        pt_time = benchmark_torch_function(100, pt_mod, input_pt)
        print(f"PT batch_size: {batch_size}, {pt_time} ms")
        with open("sd_pt_benchmark.txt", "a") as f:
            f.write(f"clip batch_size: {batch_size}, latency: {pt_time} ms\n")

    # run AIT clip
    ys = []
    num_ouputs = len(exe_module.get_output_name_to_index_map())
    print(f"num_outputs is {num_ouputs}")
    for i in range(num_ouputs):
        shape = exe_module.get_output_maximum_shape(i)
        print(f"shape is {shape}")
        ys.append(torch.empty(shape).cuda().half())
    # exe_module.run_with_tensors(inputs, ys)

    # TODO: verification
    # if verify:
    #     eps = 1e-1
    #     pt_np = pt_ys[0].detach().cpu().numpy()
    #     np.testing.assert_allclose(
    #         pt_np,
    #         ys[0].cpu().numpy(),
    #         atol=eps,
    #         rtol=eps,
    #     )
    #     print("CLIPTextTransformer verification pass")


    # AIT benchmark
    # warmup
    exe_module.benchmark_with_tensors(inputs=input_ait, outputs=ys, count=100, repeat=4)
    # benchmark
    t, a, b = exe_module.benchmark_with_tensors(inputs=input_ait, outputs=ys, count=100, repeat=4)
    print(f"output_shape: {b['output_0'].shape}")
    with open("sd_ait_benchmark.txt", "a") as f:
        f.write(f"clip batch_size: {batch_size}, latency: {t} ms\n")


@click.command()
@click.option("--batch-size", default=1, help="batch size")
@click.option("--verify", type=bool, default=False, help="verify correctness")
@click.option("--benchmark-pt", type=bool, default=True, help="run pt benchmark")
def benchmark(batch_size, verify, benchmark_pt):
    # assert batch_size == 1, "batch size must be 1 for submodule verification"
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    torch.manual_seed(4896)

    # CLIP
    benchmark_clip(batch_size=batch_size, benchmark_pt=benchmark_pt, verify=verify)


if __name__ == "__main__":
    benchmark()
