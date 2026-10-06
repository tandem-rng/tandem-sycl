# tandem-sycl

Header-only [SYCL](https://www.khronos.org/sycl/) implementation of Tandem8x32. It produces the
stream of the [specification](https://github.com/tandem-rng/spec/blob/main/SPEC.md) bit for
bit, on any SYCL device.

- [API](api.md): the fills, `tandem::Rng` and its draws.
- [Design](design.md): the fill kernels, normals and exponentials.
- [Tests](tests.md): what the suite checks and what CI runs.
- [Speed](speed.md): A100 figures against tandem-cuda and cuRAND.

## Install

Tested with AdaptiveCpp 25.10.0 (OpenMP CPU backend, and CUDA on an NVIDIA A100) and Intel
oneAPI DPC++ 2026.1.1 (OpenCL CPU runtime 2026.1.1).

Header-only. Clone with the submodule:

```sh
git clone --recurse-submodules https://github.com/tandem-rng/tandem-sycl
git submodule update --init           # in a clone made without it
```

Add `include/` and `external/tandem-cuda/include/` to the include path of a SYCL build, or use
CMake:

```cmake
add_subdirectory(tandem-sycl)
target_link_libraries(app PRIVATE tandem::sycl)
add_sycl_to_target(TARGET app)         # AdaptiveCpp, or -fsycl for DPC++
```

The `tandem::sycl` target carries the include paths and C++20, not the SYCL flags, because
SYCL is a compiler mode. With DPC++, build with `-ffp-model=precise`: `icpx` defaults to fast
math, whose float `log` and `cos` in kernels miss the normals' tolerance by about 1e-4
relative. The host normal draws use `std::fma` for bit identical results on every compiler:
on x86, build with `-mfma`, or each multiply-add is a slow library call.

`pixi.toml` provides three environments:

```sh
pixi run test                         # AdaptiveCpp 25.10, OpenMP backend on the host CPU
pixi run -e dpcpp test-dpcpp          # DPC++ 2026.1 with Intel's OpenCL CPU runtime, Linux
pixi run -e cuda test-cuda            # AdaptiveCpp 25.10 for CUDA 12.9, Linux with an NVIDIA GPU
```

AdaptiveCpp compiles kernels once to LLVM IR and translates them for the device at the first
launch (its generic SSCP target), so one binary runs on the CPU and on NVIDIA GPUs. The CUDA
build ran on driver 570, which supports CUDA 12.8, as the kernels reach the driver as PTX.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.
