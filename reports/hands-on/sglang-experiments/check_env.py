#!/usr/bin/env python3
import torch
print(f'Torch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import sglang
print(f'SGLang: {sglang.__version__}')
try:
    import flashinfer
    print(f'FlashInfer: {flashinfer.__version__}')
except:
    print('FlashInfer: not available')
