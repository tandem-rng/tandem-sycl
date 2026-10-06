<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .sycl"></p>

# tandem-sycl

[![CI](https://github.com/tandem-rng/tandem-sycl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tandem-rng/tandem-sycl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-tandem--rng.github.io-7fb3ee.svg)](https://tandem-rng.github.io/tandem-sycl/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

Header-only [SYCL](https://www.khronos.org/sycl/) implementation of
[Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic pseudorandom number
generator. It produces the stream the specification defines, bit for bit, on any SYCL device.
On the A100 its normal fills write 1160 GiB/s (f32) and 1066 GiB/s (f64 ziggurat) at the card's power cap.

It needs C++20 and AdaptiveCpp 25.10 or DPC++ 2026.1. The core comes from tandem-cuda
`0ff5f18`, a submodule in `external/tandem-cuda`.

```sh
git clone --recurse-submodules https://github.com/tandem-rng/tandem-sycl
pixi run test                         # AdaptiveCpp, OpenMP backend on the host CPU
```

```cpp
#include <tandem/sycl.hpp>

sycl::queue q;
tandem::Rng rng(42);                        // 128-bit seed as two halves, default K = 32
double *x = sycl::malloc_device<double>(n, q);
tandem::fill(q, x, n, rng).wait();          // the spec's Float64 fill, USM or sycl::buffer
tandem::Rng worker = rng.split(7);          // by index, from the key alone
tandem::fill_exponential(q, x, n, worker).wait();   // -log(1 - u), bit exact with tandem-c
```

See [API](docs/api.md) for every fill and `Rng` draw, and [design](docs/design.md),
[tests](docs/tests.md) and [speed](docs/speed.md) for the rest.

Portions of the code were generated with the assistance of LLMs.

[Documentation](https://tandem-rng.github.io/tandem-sycl/) · [Apache 2.0 license](LICENSE)
