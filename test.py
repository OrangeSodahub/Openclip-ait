import torch
from open_clip import tokenizer
import open_clip
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test_inference():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

    image = preprocess(Image.open("/home/zonlin/Jina/openclip-ait/assets/pic.jpg")).unsqueeze(0)
    text = tokenizer.tokenize(["a diagram", "a dog", "a cat"])
    print(image.shape)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    assert text_probs.cpu().numpy()[0].tolist() == [1.0, 0.0, 0.0]

test_inference()