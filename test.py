import torch
from open_clip import tokenizer
import open_clip
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_inference():
    model = open_clip.create_model('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    text = tokenizer.tokenize(["a diagram", "a dog", "a cat"])
    print(text.shape)

    with torch.no_grad():
        text_features = model.encode_text(text)
        print(text_features.shape)

test_inference()