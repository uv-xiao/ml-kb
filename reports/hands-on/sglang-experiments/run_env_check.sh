#!/bin/bash
export TMPDIR=/home/uvxiao/tmp
source /home/uvxiao/mlkb/code-repos/sglang/.venv/bin/activate
python -c "
import torch
print(f'Torch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import sglang
print(f'SGLang: {sglang.__version__}')
import flashinfer
print(f'FlashInfer: {flashinfer.__version__}')
"
