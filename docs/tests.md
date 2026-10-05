# Tests

```sh
pixi run test                         # AdaptiveCpp 25.10, OpenMP backend on the host CPU
pixi run -e dpcpp test-dpcpp          # DPC++ 2026.1 with Intel's OpenCL CPU runtime, Linux
pixi run -e cuda test-cuda            # AdaptiveCpp 25.10 for CUDA 12.9, Linux with an NVIDIA GPU
```

## Suite

`tests/test_tandem.cpp` runs every check on the default SYCL device: 3071 checks on the
AdaptiveCpp CPU device, 3581 on the A100, which also runs the float normal fill's shuffle
variant. It
checks every vector of the specification, compares fills with both kernels and in-kernel scalar
draws against reference stream dumps in `tests/data` (from tandem-cuda), compares fills against
in-kernel draws at random keys, chunk lengths, positions, lengths and output alignments, checks
fills of every type (bool, 8 to 64-bit, float, double) against the bits of the u32 stream,
checks that fills split at arbitrary points with host draws between them continue one stream,
and checks mixed-width draws, random access and derived keys in kernels. In-kernel bounded
draws and double normals equal the host's, in-kernel float normals agree with them to the
tolerance in [design](design.md). Bounded
fills are checked against Appendix A written out on host generators, against the sequential
`urand(range)` calls, and against tandem-cuda's fixtures `tests/cross_fill_below.h` at
positions 0, 1 and 12345 bits, with rejected draws. A bounded fill cut at an element boundary
must equal the whole fill at unaligned nonzero starts with ranges that reject a quarter of the
draws. Double normal fills of both kernels must equal the scalar `normal()` calls bit for bit
at random keys, chunk lengths, positions, lengths and alignments, with misses and tail values
among them. A fill cut at an odd element, at its first miss and just after must equal the
whole. They must equal tandem-cuda's `tests/cross_fill_normal.h`, and the FNV-1a hash of 10^6
normals from five starts must equal tandem-c's `a61cfa844c85f7c1`. Float normal
fills of every kernel variant are checked against the scalar `normalf2()` calls to 16 ulps plus
1e-6, from random positions and counts and from starts at every word offset within a block and
several lanes, with K from 1 to 64, and against `tests/cross_fill_normal.h`. Exponential
fills of both kernels must equal the scalar `exponential()` calls and tandem-cuda's
`tests/cross_fill_exponential.h` bit for bit, a cut fill must equal the whole, the FNV-1a hash
of 10^6 double and 10^6 float exponentials from five starts must equal tandem-c's, and 10^7
samples must match the Exp(1) law in their first four moments and a Kolmogorov-Smirnov test.
In-kernel exponentials equal the host's bit for bit. Signed, Float16, `sycl::half` and complex
fills, buffers of rank 1 and 2, and the position after empty fills at odd positions are checked
too.

## Fixtures

`tests/vectors.hpp` is generated from the spec repository's `vectors.json` by
`tools/gen_vectors.py`.

## CI

CI runs the tests on the CPU device with AdaptiveCpp on Linux and macOS and with DPC++ on
Linux, all with `-Wall -Wextra -Werror`, and fails when the vector header or the data files
differ from upstream or the submodule pin leaves tandem-cuda's main. The suite also passes on
an NVIDIA A100 with the `cuda` environment.
