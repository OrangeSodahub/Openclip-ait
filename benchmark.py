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


def final_projection(x, text, pt_mod):
    """
    This operation is extracted from CLIPTextTransformer.encode_text()
    need to rewrite in ait and move back to original model
    
    : param x: shape (batch_size, seqlen, hidden)
    : param text: shape (1, batch_size, seqlen)
    """
    x = x['output_0']
    params = dict(pt_mod.named_parameters())
    for key, arr in params.items():
        if key == 'text_projection':
            if arr.dtype == torch.float32:
                arr.data = arr.data.half()
            return x[torch.arange(x.shape[0]), text[0].argmax(dim=-1)] @ arr


def benchmark_clip(
    runtime_path,
    batch_size=1,
    mode="text",
    benchmark_pt=False,
    save_results=False,
):
    # set up models
    exe_module = Model(runtime_path)
    if exe_module is None:
        print("Error!! Cannot find compiled module for CLIPTextModel.")
        exit(-1)

    openclip_mod = OpenCLIPModel(name='ViT-L-14::laion2b-s32b-b82k', device='cuda')
    pt_mod = openclip_mod._model
    pt_mod = pt_mod.eval()

    # set up inputs
    # TODO: wrong inputs
    text = tokenizer.tokenize(["a diagram"]).cuda()
    preprocess = image_transform(224, is_train=False)
    if mode == "text":
        input_ait = torch.randint(0, 10, (1, batch_size, 77), dtype=torch.int64).long().cuda()
        input_pt = input_ait[0]
    elif mode == "vision":
        input = torch.randint(0, 10, (1, batch_size, 224, 224, 3), dtype=torch.int64)
        input_ait = input.half().cuda()
        input_pt = input[0].permute((0, 3, 1, 2)).cuda()

    # TODO: attention mask

    # PT benchmark
    if benchmark_pt:
        pt_time = benchmark_torch_function(100, pt_mod, input_pt)
        print(f"PT batch_size: {batch_size}, {pt_time} ms")
        with open("sd_pt_benchmark.txt", "a") as f:
            f.write(f"clip batch_size: {batch_size}, latency: {pt_time} ms\n")
        res_pt = pt_mod(input_pt)

    # AIT benchmark
    ys = []
    num_ouputs = len(exe_module.get_output_name_to_index_map())
    print(f"num_outputs is {num_ouputs}")
    for i in range(num_ouputs):
        shape = exe_module.get_output_maximum_shape(i)
        print(f"shape is {shape}")
        ys.append(torch.empty(shape).cuda().half())

    exe_module.benchmark_with_tensors(inputs=input_ait, outputs=ys, count=100, repeat=4) # warm up
    t, a, b = exe_module.benchmark_with_tensors(inputs=input_ait, outputs=ys, count=100, repeat=4)
    if mode == 'text':
        res_ait = final_projection(b, input_ait, pt_mod)
    elif mode == 'vision':
        res_ait = b['output_0']
    print(f"output_shape: {res_ait.shape}")

    # output results
    res_pt = res_pt.cpu().detach().numpy()
    res_ait = res_ait.cpu().detach().numpy()
    # ----------------------------------------- debug ---------------------------------------------------
    if save_results:
        import numpy as np
        np.savetxt("./test/res_pt.txt", res_pt[0])
        np.savetxt("./test/res_ait.txt", res_ait[0])
    # ---------------------------------------------------------------------------------------------------
    max_diff = abs(res_pt-res_ait).max()
    mean_diff = abs(res_pt-res_ait).mean()
    print(f"{max_diff=:.5f}")
    print(f"{mean_diff=:.5f}")

    with open("sd_ait_benchmark.txt", "a") as f:
        f.write(f"clip batch_size: {batch_size}, latency: {t} ms\n")


def benchmark(batch_size, benchmark_pt, mode):
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    torch.manual_seed(4896)

    if mode == "text":
        path = "./CLIPModel/CLIPTextModel/test.so"
    else:
        path = "./CLIPModel/CLIPVisionModel/test.so"
    # CLIP
    benchmark_clip(
        runtime_path=path,
        batch_size=batch_size,
        mode=mode,
        benchmark_pt=benchmark_pt,
        save_results=False,
    )


if __name__ == "__main__":
    benchmark(
        batch_size=1,
        benchmark_pt=True,
        mode="vision",
    )
