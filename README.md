# Openclip-ait

<<<<<<< HEAD
Tested on RTX3080:
Model: `ViT-L-14::laion2b-s32b-b82k` on RTX3080:

-----------------------------------------------------------------------------------
shape           | pt (ms)  | ait (ms) | without `flash_attn` | mean idff | max diff
----------------|----------|----------|--------------------------------------------
(1, 77)         |8.6888    |0.8523    |1.6269                |0.00335    |0.01758
(2, 77)         |8.7543    |0.9854    |2.0161                |0.00333    |0.01782
(4, 77)         |8.7231    |1.2459    |2.8970                |0.00358    |0.04297
(8, 77)         |9.4466    |2.0201    |4.8552                |0.00355    |0.03906
(16, 77)        |10.0222   |3.4399    |8.7880                |0.00333    |0.03906
(1, 224, 224, 3)|18.0799   |3.7753    |8.4608                |           |
(2, 224, 224, 3)|17.9421   |          |8.4604                |           |

Model: `ViT-g-14::laion2b-s12b-b42k`

-----------------------------------------------------------------------------------
shape           | pt (ms)  | ait (ms) | without `flash_attn` | mean idff | max diff
----------------|----------|----------|--------------------------------------------
(1, 77)         |-         |          |-                     |           |
(1, 224, 224, 3)|30.3925   |          |13.8009               |           |
=======
Tested on RTX3080

----------------------------------------------------------------------------------------------------------
shape     | model                      | pt (ms)  | ait (ms) | without `flash_attn` | mean idff | max diff
----------|----------------------------|----------|----------|----------------------|-----------|---------
(1, 77)   |ViT-L-14::laion2b-s32b-b82k |8.6888    |0.8523    |1.6269                |0.00335    |0.01758
(2, 77)   |                            |8.7543    |0.9854    |                      |           |
(4, 77)   |                            |8.7231    |1.2459    |                      |           |
(8, 77)   |                            |9.4466    |2.0201    |                      |           |
(16, 77)  |                            |10.0222   |3.4399    |                      |           |
(1, 224, 224, 3)|                      |16.2587   |3.7753    |5.3559                |           |
(2, 224, 224, 3)|                      |17.9421   |          |8.4604                |           |
(1, 77)   |ViT-g-14::laion2b-s12b-b42k |-         |          |-                     |           |
(1, 224, 224, 3)|                      |30.3925   |          |13.8009               |           |
>>>>>>> dcbaa7b6b20fa3d3f504e552dc84b63198ec9f69


## Known Issues:
- Index Tensor with Tensor not supported in `encode_text` (see: https://github.com/facebookincubator/AITemplate/issues/49)
- Vit-g-14 with head_size=88 not supported by flash attention (see: https://github.com/facebookincubator/AITemplate/issues/53)


## Reference
- AITemplate: https://github.com/facebookincubator/aitemplate/
- Openclip: https://github.com/mlfoundations/open_clip
- Openai/clip: https://github.com/openai/CLIP
