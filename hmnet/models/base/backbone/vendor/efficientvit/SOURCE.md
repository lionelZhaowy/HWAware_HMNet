# EfficientViT upstream snapshot

Source: https://github.com/mit-han-lab/efficientvit; local official checkout `efficientvit-master-hanlab`.
Import namespaces and lazy loading of unused Triton normalization were adapted; the unused classification-training helper `reset_bn` (and its external apps import) was omitted; B1 operators are unchanged. No training framework is vendored.

Original file SHA256:

- `models/efficientvit/backbone.py`: `027a0bad8beea923d0992d3d75a9033bdd14e7e3fdb1e60f9b6561b250b40ccd`
- `models/nn/ops.py`: `6e632634ae60eb6b7fb738b393e7a111bc80e2c074cc4f527fb75e13952fffbe`
- `models/nn/act.py`: `4bcf83cb4e28c2b83e000b244948d143feb7e3784e67407fda383604abc6074f`
- `models/nn/norm.py`: `efaad2af9fd6805e18c759405b534a9fbb0be39a8e952a7a3c1c52c371f8bdef`
- `models/nn/triton_rms_norm.py`: `b9ccc6000a3d6cadd0f43614ffe4590cc0bc4838c81de9ad291175a0053b8033`
- `models/utils/list.py`: `de0b3b62f0f7eebe6f83679697ea5955a6a502eaae5695bb86c0c0340a6e778e`
- `models/utils/network.py`: `b437a1f41c7721e47def883b4e1a8b142bfc295372007ffc0b358bd1d0dfc6aa`
- `models/utils/random.py`: `6b0428cf88fc6feb5abdb3de4276fe5db4984d1d5a0a76750122af13c14643d7`
- `LICENSE`: `3c3976a8a7a11c899b2dcdf71cb61198cb302e9542cd99cdad5f1d5d0afe3803`
