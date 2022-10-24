# Openclip-ait
## Tested on RTX3080
------------------------------------------------------------
shape     | model                      | pt (ms)  | ait (ms) 
----------|----------------------------|----------|----------
(1, 77)   |ViT-B-16::laion400m_e31     |8.5705    |0.6386
(2, 77)   |                            |8.8310    |0.7083
(4, 77)   |                            |8.7231    |0.7927
(1, 77)   |ViT-L-14::laion400m_e31     |8.5237    |0.8651
(2, 77)   |                            |8.7861    |0.9946
(4, 77)   |                            |8.9864    |1.2536
(8, 77)   |                            |9.6915    |2.0272
(16, 77)  |                            |10.3835   |3.4425
(1, 224, 224, 3)|                      |17.0629   |3.7626
(1, 77)   |ViT-L-14::laion2b-s32b-b82k |8.6888    |0.8523
(2, 77)   |                            |8.7543    |0.9854
(4, 77)   |                            |8.7231    |1.2459
(8, 77)   |                            |9.4466    |2.0201
(16, 77)  |                            |10.0222   |3.4399
(1, 224, 224, 3)|                      |16.2587   |3.7753
------------------------------------------------------------

## Known Issues:
- ViT-g-14 not supported: `head_size` only be [8, 16, 32, 64, 128]
(see: https://github.com/facebookincubator/AITemplate/issues/53)
- `batch_size` > 1 not supported in VisionTransformer (see: https://github.com/facebookincubator/AITemplate/blob/44026ba7e7f5376a80cf0f2b333a0f25c0eeda6c/python/aitemplate/backend/cuda/tensor/expand.py#L21 ) (still need to verify.)
- Index Tensor with Tensor not supported in `encode_text` (see: https://github.com/facebookincubator/AITemplate/issues/49)
