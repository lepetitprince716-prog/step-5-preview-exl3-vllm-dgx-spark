"""Build step5_spark.tf_exl3._C, TensorFold's grouped EXL3 routed-experts kernels, for one GPU architecture.

    TORCH_CUDA_ARCH_LIST=12.1 python3 setup.py build_ext --inplace --build-temp /tmp/tf_exl3_build

Run it inside the serving image (build.sh does) so the module matches the server's torch and CUDA. It needs nvcc and a
C++ compiler, not a GPU and not ninja: TORCH_CUDA_ARCH_LIST names the architecture instead of a device query.

The serving image's /usr/local/cuda carries nvcc without the cuSPARSE headers torch's ATen/cuda includes and without
libcudart.so; both come from NVIDIA's pip CUDA beside torch. Its headers are searched after CUDA_HOME's (-isystem), so
nvcc's own headers win, and libcudart.so is a symlink to the pip runtime torch itself loads.

experts_bind.cpp is upstream's experts.cpp, renamed: setuptools names objects by stem, so experts.cpp and experts.cu
would both compile to experts.o and the second would replace the first (torch's JIT build names them apart).
"""
import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    raise SystemExit("set TORCH_CUDA_ARCH_LIST (12.1 for GB10) so the build targets one architecture")

SOURCES = ["experts_bind.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu"]
LINKS = Path(os.environ.get("TF_EXL3_LINKS", "/tmp/tf_exl3_links"))


def pip_cuda() -> Path:
    return Path(torch.__file__).resolve().parents[1] / "nvidia" / f"cu{torch.version.cuda.split('.')[0]}"


def toolkit_flags() -> tuple[list[str], list[str]]:
    """(-isystem for the missing headers, -L for a bare libcudart.so), each empty when CUDA_HOME has them."""
    pip = pip_cuda()
    inc: list[str] = []
    if not Path(CUDA_HOME, "include", "cusparse.h").exists():
        if not (pip / "include" / "cusparse.h").exists():
            raise SystemExit(f"cusparse.h is in neither {CUDA_HOME}/include nor {pip}/include")
        inc = ["-isystem", str(pip / "include")]
    link: list[str] = []
    if not Path(CUDA_HOME, "lib64", "libcudart.so").exists():
        versioned = sorted((pip / "lib").glob("libcudart.so.*"))
        if not versioned:
            raise SystemExit(f"libcudart.so is in neither {CUDA_HOME}/lib64 nor {pip}/lib")
        LINKS.mkdir(parents=True, exist_ok=True)
        bare = LINKS / "libcudart.so"
        if bare.is_symlink() or bare.exists():
            bare.unlink()
        bare.symlink_to(versioned[0])
        link = [f"-L{LINKS}"]
    return inc, link


INC, LINK = toolkit_flags()

setup(
    name="step5_spark_tf_exl3",
    ext_modules=[CUDAExtension(name="_C", sources=SOURCES,
                               extra_compile_args={"cxx": ["-O3", *INC], "nvcc": ["-O3", "-lineinfo", *INC]},
                               extra_link_args=LINK)],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
