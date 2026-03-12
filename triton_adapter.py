from __future__ import annotations

"""
Utilities for plugging official Triton tutorial kernels into the noise-aware
benchmarking engine without executing the tutorials' top-level demo code.

Primary sources:
- https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html

Expected workflow on a GPU-capable machine:
1. Install Triton (`pip install triton`) and a CUDA/HIP-capable PyTorch build.
2. Place the official tutorial files under `kernel_noise_prototype/triton_tutorials/`
   or point environment variables at custom locations.
3. Run `python kernel_noise_prototype/setup_gpu.py`.
4. Run `python kernel_noise_prototype/demo.py --backend triton --workload matmul`
   or `--workload softmax`.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Sequence

import torch


@dataclass
class TritonEnvironment:
    tutorial_path: Path
    module: ModuleType
    device: str
    workload: str


@dataclass(frozen=True)
class KernelVariantSpec:
    name: str
    block_size_m: int
    block_size_n: int
    block_size_k: int
    group_size_m: int
    num_warps: int
    num_stages: int
    activation: str = ""


def default_matmul_variant_specs() -> list[KernelVariantSpec]:
    # These are a small, explicit subset of the official Triton tutorial
    # autotune search space. They give us a credible "generated candidates"
    # layer without requiring a full kernel synthesis system.
    return [
        KernelVariantSpec(
            name="generated_matmul_balanced_128x128x64_w4_s4",
            block_size_m=128,
            block_size_n=128,
            block_size_k=64,
            group_size_m=8,
            num_warps=4,
            num_stages=4,
        ),
        KernelVariantSpec(
            name="generated_matmul_wide_n_64x256x32_w4_s4",
            block_size_m=64,
            block_size_n=256,
            block_size_k=32,
            group_size_m=8,
            num_warps=4,
            num_stages=4,
        ),
        KernelVariantSpec(
            name="generated_matmul_aggressive_128x256x64_w8_s3",
            block_size_m=128,
            block_size_n=256,
            block_size_k=64,
            group_size_m=8,
            num_warps=8,
            num_stages=3,
        ),
        KernelVariantSpec(
            name="generated_matmul_compact_64x64x32_w2_s5",
            block_size_m=64,
            block_size_n=64,
            block_size_k=32,
            group_size_m=8,
            num_warps=2,
            num_stages=5,
        ),
    ]


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
        return True
    except Exception:
        return False


def _default_tutorial_dir() -> Path:
    return Path(__file__).resolve().parent / "triton_tutorials"


def default_matmul_tutorial_path() -> Path:
    env_path = os.environ.get("TRITON_TUTORIAL_MATMUL")
    if env_path:
        return Path(env_path)
    return _default_tutorial_dir() / "03-matrix-multiplication.py"


def default_softmax_tutorial_path() -> Path:
    env_path = os.environ.get("TRITON_TUTORIAL_SOFTMAX")
    if env_path:
        return Path(env_path)
    return _default_tutorial_dir() / "02-fused-softmax.py"


def default_tutorial_path() -> Path:
    return default_matmul_tutorial_path()


def _require_tutorial(path: Path, env_var_name: str) -> None:
    if path.exists():
        return
    raise FileNotFoundError(
        "Triton tutorial file not found. Place the official tutorial at "
        f"`{path}` or set {env_var_name}."
    )


def _load_partial_tutorial_module(
    tutorial_path: Path,
    module_name: str,
    stop_markers: list[str],
) -> ModuleType:
    source = tutorial_path.read_text(encoding="utf-8")
    cut_index = len(source)
    for marker in stop_markers:
        idx = source.find(marker)
        if idx != -1 and idx < cut_index:
            cut_index = idx
    trimmed = source[:cut_index]
    if not trimmed.strip():
        raise RuntimeError(f"Failed to extract reusable Triton code from {tutorial_path}")

    module = ModuleType(module_name)
    module.__file__ = str(tutorial_path)
    exec(compile(trimmed, str(tutorial_path), "exec"), module.__dict__)
    return module


def load_matmul_environment(device: str = "cuda") -> TritonEnvironment:
    tutorial_path = default_matmul_tutorial_path()
    _require_tutorial(tutorial_path, "TRITON_TUTORIAL_MATMUL")
    if not triton_available():
        raise RuntimeError("Triton is not installed in this environment.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available for Triton demo.")

    module = _load_partial_tutorial_module(
        tutorial_path=tutorial_path,
        module_name="triton_tutorial_matmul",
        stop_markers=[
            "torch.manual_seed(0)",
            "benchmark.run(",
        ],
    )
    if not hasattr(module, "matmul"):
        raise AttributeError("Loaded Triton matmul tutorial does not expose `matmul`.")

    return TritonEnvironment(
        tutorial_path=tutorial_path,
        module=module,
        device=device,
        workload="matmul",
    )


def load_softmax_environment(device: str = "cuda") -> TritonEnvironment:
    tutorial_path = default_softmax_tutorial_path()
    _require_tutorial(tutorial_path, "TRITON_TUTORIAL_SOFTMAX")
    if not triton_available():
        raise RuntimeError("Triton is not installed in this environment.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available for Triton demo.")

    module = _load_partial_tutorial_module(
        tutorial_path=tutorial_path,
        module_name="triton_tutorial_softmax",
        stop_markers=[
            "torch.manual_seed(0)",
            "benchmark.run(",
        ],
    )
    if not hasattr(module, "softmax"):
        raise AttributeError("Loaded Triton softmax tutorial does not expose `softmax`.")

    return TritonEnvironment(
        tutorial_path=tutorial_path,
        module=module,
        device=device,
        workload="softmax",
    )


def load_tutorial_environment(device: str = "cuda", workload: str = "matmul") -> TritonEnvironment:
    if workload == "matmul":
        return load_matmul_environment(device=device)
    if workload == "softmax":
        return load_softmax_environment(device=device)
    raise ValueError(f"Unsupported Triton workload: {workload}")


def build_triton_matmul_candidates(
    m: int = 512,
    k: int = 512,
    n: int = 512,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> Dict[str, Callable[[], None]]:
    env = load_matmul_environment(device=device)

    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)

    def torch_cuda_matmul_baseline() -> None:
        torch.matmul(a, b)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def triton_tutorial_matmul() -> None:
        env.module.matmul(a, b)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def triton_tutorial_matmul_noisy() -> None:
        env.module.matmul(a, b)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    return {
        "torch_cuda_matmul_baseline": torch_cuda_matmul_baseline,
        "triton_tutorial_matmul": triton_tutorial_matmul,
        "triton_tutorial_matmul_noisy": triton_tutorial_matmul_noisy,
    }


def build_generated_triton_matmul_candidates(
    m: int = 512,
    k: int = 512,
    n: int = 512,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    variant_specs: Sequence[KernelVariantSpec] | None = None,
) -> Dict[str, Callable[[], None]]:
    env = load_matmul_environment(device=device)
    specs = list(variant_specs or default_matmul_variant_specs())

    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)

    def torch_cuda_matmul_baseline() -> None:
        torch.matmul(a, b)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def make_variant(spec: KernelVariantSpec) -> Callable[[], None]:
        def run() -> None:
            c = torch.empty((m, n), device=device, dtype=torch.float16)
            grid = (
                lambda meta: (
                    env.module.triton.cdiv(m, meta["BLOCK_SIZE_M"])
                    * env.module.triton.cdiv(n, meta["BLOCK_SIZE_N"]),
                )
            )
            env.module.matmul_kernel[grid](
                a,
                b,
                c,
                m,
                n,
                k,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                BLOCK_SIZE_M=spec.block_size_m,
                BLOCK_SIZE_N=spec.block_size_n,
                BLOCK_SIZE_K=spec.block_size_k,
                GROUP_SIZE_M=spec.group_size_m,
                ACTIVATION=spec.activation,
                num_warps=spec.num_warps,
                num_stages=spec.num_stages,
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()

        return run

    candidates: Dict[str, Callable[[], None]] = {
        "torch_cuda_matmul_baseline": torch_cuda_matmul_baseline,
    }
    for spec in specs:
        candidates[spec.name] = make_variant(spec)

    # Include one explicit noisy candidate so the benchmark engine still has a
    # superficially-fast but unstable finalist to reason about.
    if specs:
        noisy_spec = specs[0]
        candidates[f"{noisy_spec.name}_noisy"] = make_variant(noisy_spec)

    return candidates


def build_triton_softmax_candidates(
    rows: int = 4096,
    cols: int = 1024,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> Dict[str, Callable[[], None]]:
    env = load_softmax_environment(device=device)

    x = torch.randn((rows, cols), device=device, dtype=dtype)

    def torch_cuda_softmax_baseline() -> None:
        torch.softmax(x, dim=-1)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def triton_tutorial_softmax() -> None:
        env.module.softmax(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def triton_tutorial_softmax_noisy() -> None:
        env.module.softmax(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    return {
        "torch_cuda_softmax_baseline": torch_cuda_softmax_baseline,
        "triton_tutorial_softmax": triton_tutorial_softmax,
        "triton_tutorial_softmax_noisy": triton_tutorial_softmax_noisy,
    }


def build_triton_candidates(
    workload: str = "matmul",
    device: str = "cuda",
    candidate_mode: str = "tutorial",
) -> Dict[str, Callable[[], None]]:
    if workload == "matmul":
        if candidate_mode == "tutorial":
            return build_triton_matmul_candidates(device=device)
        if candidate_mode == "generated":
            return build_generated_triton_matmul_candidates(device=device)
        raise ValueError(f"Unsupported matmul candidate mode: {candidate_mode}")
    if workload == "softmax":
        if candidate_mode != "tutorial":
            raise ValueError("Softmax currently supports only tutorial candidate mode.")
        return build_triton_softmax_candidates(device=device)
    raise ValueError(f"Unsupported Triton workload: {workload}")
