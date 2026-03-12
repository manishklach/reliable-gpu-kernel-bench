from __future__ import annotations

import shutil
import subprocess
import sys

import torch

from triton_adapter import (
    default_matmul_tutorial_path,
    default_softmax_tutorial_path,
    load_matmul_environment,
    load_softmax_environment,
    triton_available,
)


def print_line(label: str, value: str) -> None:
    print(f"{label}: {value}")


def main() -> int:
    print("GPU setup check for kernel_noise_prototype")
    print()

    print_line("Python", sys.version.split()[0])
    print_line("PyTorch", torch.__version__)
    print_line("torch.cuda.is_available()", str(torch.cuda.is_available()))
    print_line("Triton importable", str(triton_available()))
    print_line("Matmul tutorial path", str(default_matmul_tutorial_path()))
    print_line("Matmul tutorial exists", str(default_matmul_tutorial_path().exists()))
    print_line("Softmax tutorial path", str(default_softmax_tutorial_path()))
    print_line("Softmax tutorial exists", str(default_softmax_tutorial_path().exists()))

    nvidia_smi = shutil.which("nvidia-smi")
    print_line("nvidia-smi", nvidia_smi or "not found")
    if nvidia_smi:
        try:
            output = subprocess.check_output(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem",
                    "--format=csv,noheader",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
            print_line("GPU info", output.splitlines()[0] if output else "no output")
        except Exception as exc:
            print_line("GPU info", f"query failed: {exc}")

    if triton_available() and default_matmul_tutorial_path().exists():
        try:
            env = load_matmul_environment()
            print_line("Matmul tutorial import", f"ok ({env.tutorial_path.name})")
            print_line("Matmul tutorial module", env.module.__name__)
        except Exception as exc:
            print_line("Matmul tutorial import", f"failed: {exc}")
            return 1
    else:
        print_line(
            "Matmul tutorial import",
            "skipped (requires Triton installed and tutorial file present)",
        )

    if triton_available() and default_softmax_tutorial_path().exists():
        try:
            env = load_softmax_environment()
            print_line("Softmax tutorial import", f"ok ({env.tutorial_path.name})")
            print_line("Softmax tutorial module", env.module.__name__)
        except Exception as exc:
            print_line("Softmax tutorial import", f"failed: {exc}")
            return 1
    else:
        print_line(
            "Softmax tutorial import",
            "skipped (requires Triton installed and tutorial file present)",
        )

    print()
    print("If all checks look good, run:")
    print("python kernel_noise_prototype/demo.py --backend triton --workload matmul")
    print("python kernel_noise_prototype/demo.py --backend triton --workload matmul --candidate-mode generated")
    print("python kernel_noise_prototype/demo.py --backend triton --workload softmax")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
