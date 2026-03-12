# Moving This Prototype to a GPU Machine

## 1. Copy the folder

Copy the entire `kernel_noise_prototype` folder to the target GPU machine.

## 2. Install dependencies

```powershell
python -m pip install -r kernel_noise_prototype/requirements-gpu.txt
```

Note:
- install a CUDA/HIP-capable PyTorch build appropriate for the target machine
- `triton` should be installed in the same environment

## 3. Triton tutorial kernels already included

These files are already present in the folder:

- `kernel_noise_prototype/triton_tutorials/03-matrix-multiplication.py`
- `kernel_noise_prototype/triton_tutorials/02-fused-softmax.py`

If you move the whole folder, no extra download is needed.

Optional:
- set `TRITON_TUTORIAL_MATMUL` if you want to use a different matrix-multiplication tutorial path
- set `TRITON_TUTORIAL_SOFTMAX` if you want to use a different fused-softmax tutorial path

## 4. Run in Triton mode

```powershell
python kernel_noise_prototype/demo.py --backend triton --workload matmul
python kernel_noise_prototype/demo.py --backend triton --workload softmax
```

## 5. Expected behavior

The matmul mode will try to build these real candidates:
- `torch_cuda_matmul_baseline`
- `triton_tutorial_matmul`
- `triton_tutorial_matmul_noisy`

The softmax mode will try to build these real candidates:
- `torch_cuda_softmax_baseline`
- `triton_tutorial_softmax`
- `triton_tutorial_softmax_noisy`

The decision engine remains the same:
- repeated trials
- telemetry capture
- contamination classification
- finalist-pair reruns
- promote / reject / defer
