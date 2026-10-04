<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .sycl"></p>

# tandem-sycl

[SYCL](https://www.khronos.org/sycl/) implementation of
[Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic pseudorandom number
generator built to be fast on CPUs and GPUs alike. Two headers, C++20, no compiled library. It
produces the stream the specification defines, bit for bit, on any SYCL device. Tested with
AdaptiveCpp 25.10.0 (OpenMP CPU backend, and CUDA on an NVIDIA A100) and Intel oneAPI DPC++
2026.1.1 (OpenCL CPU runtime 2026.1.1).

- `tandem::fill(queue, ptr, n, rng)` and `tandem::fill(queue, buffer, rng)`: fill USM memory or a
  whole `sycl::buffer` of any rank with the draws that start at the generator's position, as the
  specification's fill defines, and move the position past them. The value type is `bool`, an
  integer of 8 to 64 bits, `float`, `double`, `sycl::half`, `std::complex<float>` or
  `std::complex<double>`. A buffer fills in its linear order. The host generator advances
  exactly as a CPU fill would, so host and device draws interleave on one stream.
- `tandem::fill_f16_bits(queue, ptr, n, rng)`: the binary16 bit patterns of the Float16 draws
  in `uint16_t`. A `uint16_t` output in `fill` gets raw 16-bit draws.
- `tandem::fill_below(queue, ptr, n, rng, range)`: uniform integers on `[0, range)` in an
  unsigned 32- or 64-bit integer type, by Lemire's method as `Rng::urand(range)`. Element `i`
  takes draw `i` of the u32 (u64) fill and consumes exactly that draw, so the fill advances the
  position by 32 n (64 n) bits whatever the draws are. A rejected draw retries on a fallback
  stream, `split(g)` of `sub(PURPOSE_BELOW32)` (or `64`) of the key at position 0, where `g`
  is the draw's global index, the aligned start over 32 (64) plus `i`. A fill without
  rejections equals the sequential `urand(range)` calls, the rare rejection costs no
  coordination, and a fill cut into pieces equals the whole fill.
- `tandem::fill_normal(queue, ptr, n, rng)`: standard normals in `float` or `double` by
  Box-Muller, the flattened sequence of `Rng::normalf2` or `Rng::normal2` calls. Pair `j`, the
  elements `2j` and `2j + 1` with the cos half first, comes from the draws `2j` and `2j + 1` of
  the Float32 (Float64) fill. An odd count drops the last sin half and still consumes both
  draws, 64 (128) bits per pair.
- `tandem::Rng`: a value type for draws inside kernels. It holds the transport form (128-bit
  key, 64-bit bit position, chunk length `K`) and one cached chunk state, 80 bytes, and a kernel
  captures it by value. Each work item takes its own generator with `rng.split(i)` or by
  position, or reads element `i` of a fill directly with `rng.at_urand(i)` and its siblings.
- `tandem/core.hpp`: the specification's building blocks and `Rng` in portable C++17 without
  SYCL or CUDA types. It comes from [tandem-cuda](https://github.com/tandem-rng/tandem-cuda),
  a git submodule in `external/tandem-cuda`, which tandem-kokkos shares.

Every fill returns a `sycl::event` and does not wait. The USM forms take an optional
`std::vector<sycl::event>` of dependencies as their last argument. The buffer forms take the
buffer's dependencies from the SYCL runtime. Positions move as in tandem-cuda: an empty
`fill` aligns the position to the type's width, and an empty `fill_below` or `fill_normal`
consumes no draws and leaves the position alone. A fill that would run past stream position
2^64 throws `std::overflow_error` before it moves the position.

## Use

```cpp
#include <tandem/sycl.hpp>

sycl::queue q;
tandem::Rng rng(42);                        // 128-bit seed as two halves, default K = 32
double *x = sycl::malloc_device<double>(n, q);
tandem::fill(q, x, n, rng);                 // the spec's Float64 fill
double u = rng.drand();                     // continues the stream after the fill

sycl::buffer<std::complex<float>, 2> z(sycl::range<2>(64, 64));
tandem::fill(q, z, rng);                    // real then imaginary part per value

uint32_t *die = sycl::malloc_device<uint32_t>(n, q);
float *g = sycl::malloc_device<float>(n, q);
auto e1 = tandem::fill_below(q, die, n, rng, 6u);   // uniform on [0, 6)
auto e2 = tandem::fill_normal(q, g, n, rng);        // standard normals

float *y = sycl::malloc_device<float>(m, q);
q.parallel_for(sycl::range<1>(m), {e1, e2}, [=](sycl::id<1> i) {
    tandem::Rng r = rng.split(i);           // one generator per work item, from the key alone
    y[i] = r.frand() + r.frand();
}).wait();
```

`Rng` draws:

| | |
|---|---|
| `bit()`, `urand()`, `urand64()`, `frand()`, `drand()` | the specification's Bool, UInt32, UInt64, Float32 and Float64 draws |
| `at_urand(i)`, `at_urand64(i)`, `at_frand(i)`, `at_drand(i)` | element `i` of the fill that would start here, without advancing |
| `urand(range)`, `urand64(range)`, `rand(start, end)`, `rand64(start, end)`, `frand(range)`, `drand(start, end)`, ... | bounded draws, uniform by Lemire's multiply and reject |
| `normal()`, `normalf()`, `normal(mean, sd)` | the cos half of a Box-Muller step from two Float64 or Float32 draws |
| `normal2()`, `normalf2()` | both halves of the step as a pair `z0`, `z1` |
| `split(i)`, `sub(purpose)`, `fork(children, n)` | child generators as the specification defines them |
| `key()`, `position()`, `set_position(p)`, `chunk_length()` | transport form |

Signed integers hold the two's complement of the unsigned draw of the same width. A complex
value takes two draws, the real and then the imaginary component.

Bounded and normal draws follow Appendix A of the specification, the contract in `core.hpp`,
and the same fills in tandem-c, tandem-cuda and tandem-kokkos. `Rng::normal` and its siblings
take the polynomial Box-Muller of `core.hpp`, the arithmetic of tandem-c's host fills, on the
host and in kernels. The double normal fill takes the same arithmetic inline, with fused
multiply-adds and IEEE division and square root, so it equals `Rng::normal2` and tandem-cuda's
fills bit for bit on every tested device. The float normal fill takes the logarithm of that
arithmetic and the fast `sycl::native::cos` and `sin`, as tandem-cuda takes `__sincosf`, with
the angle shifted by half a turn into `[-pi, pi)`. It needs no double precision on the device.
Float normals agree with the other ports to 16 ulps plus 1e-6. Define
`TANDEM_PRECISE_F32_NORMAL` for `box_muller2_f32` from `core.hpp` instead. Double fills and
`Rng::normal` need a device with `aspect::fp64`, `Rng::normalf` does not.

## Kernels

One work item steps one chunk. On GPUs and other accelerators with `K >= 8` a work group of
256 items stages eight steps of 32 groups in 32 KiB of local memory and writes them as 512
contiguous bytes per 32 items. On CPU devices, and for smaller `K`, each item stores its blocks
directly. Both store whole 16-byte blocks when the output's blocks are 16-byte aligned, which a
kernel checks on the device, so USM and buffer outputs share one path. A bool block is 128
bytes and leaves as eight 16-byte stores.

On GPUs the normal fill cuts the stream into units of four 32-bit words from the fill's first
word, two float pairs or one double pair, each one 16-byte store. Each work item steps only
its own chunk, and the eight lanes of a group write eight consecutive units, 128 aligned bytes,
one step after their rows are out. When the start is not a multiple of 32 words, a unit's
words lie in two blocks held by other lanes, which pass them by sub-group shuffles. The units
that end in the next group take its first row, kept from step 0, or for the group after the
work group computed in one step. So any start costs at most about a tenth more. The word
offset within a block is a template parameter, four kernels per type. Devices whose sub-group
sizes are not all multiples of eight pass the blocks through local memory with a barrier per
step instead. On CPU devices each item writes the pairs that end in its blocks, stepping the
previous chunk as well at such starts.

## Build

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

## Tests

`tests/test_tandem.cpp` runs every check on the default SYCL device: 4410 checks on the CPU
devices, 5676 on the A100, which also runs the normal fill's shuffle variant. It
checks every vector of the specification, compares fills with both kernels and in-kernel scalar
draws against reference stream dumps in `tests/data` (from tandem-cuda), compares fills against
in-kernel draws at random keys, chunk lengths, positions, lengths and output alignments, checks
fills of every type (bool, 8 to 64-bit, float, double) against the bits of the u32 stream,
checks that fills split at arbitrary points with host draws between them continue one stream,
and checks mixed-width draws, random access and derived keys in kernels. In-kernel bounded
draws equal the host's, in-kernel normals agree with them to the tolerance above. Bounded
fills are checked against Appendix A written out on host generators, against the sequential
`urand(range)` calls, and against tandem-cuda's fixtures `tests/cross_fill_below.h` at
positions 0, 1 and 12345 bits, with rejected draws. A bounded fill cut at an element boundary
must equal the whole fill at unaligned nonzero starts with ranges that reject a quarter of the
draws. Normal fills of every kernel variant are checked against the scalar `normal2()` calls,
bit for bit for double and to 16 ulps plus 1e-6 for float, from random positions and counts
and from starts at every word offset within a block and several lanes, with K from 1 to 64,
and against `tests/cross_fill_normal.h` from tandem-cuda (bit for bit for double). Signed,
Float16, `sycl::half` and complex fills, buffers of rank 1 and 2, and the position after empty
fills at odd positions are checked too.

`tests/vectors.hpp` is generated from the spec repository's `vectors.json` by
`tools/gen_vectors.py`. CI runs the tests on the CPU device with AdaptiveCpp on Linux and macOS
and with DPC++ on Linux, all with `-Wall -Wextra -Werror`, and fails when the vector header or
the data files differ from upstream or the submodule pin leaves tandem-cuda's main. The suite
also passes on an NVIDIA A100 with the `cuda` environment.

## Speed

NVIDIA A100 40 GB (PCIe), driver 570, AdaptiveCpp 25.10.0 for CUDA 12.9 (`pixi run -e cuda
bench-cuda`): 2^28 elements into USM device memory, the minimum of 15 runs of eight
back-to-back fills after a half-second warm-up. The GPU had no other process for 30 seconds
before the run and none during it. The tandem-cuda column is its README's figure on the same
card. Narrow types write fewer bytes for the same element count.

| GiB/s written | tandem-sycl | tandem-cuda |
|---|---|---|
| `fill` `uint32_t`, tile kernel (default on GPUs for K >= 8) | 1371 | 1386 |
| `fill` `uint64_t`, tile kernel | 1390 | 1394 |
| `fill` `float`, tile kernel | 1379 | 1381 |
| `fill` `double`, tile kernel | 1389 | 1392 |
| `fill` `uint32_t`, chunk kernel | 1318 | 1308 |
| `fill` `uint64_t`, chunk kernel | 1314 | 1315 |
| `fill` `float`, chunk kernel | 1312 | 1312 |
| `fill` `double`, chunk kernel | 1314 | 1323 |
| `fill` `uint16_t` | 1362 | 1370 |
| `fill` `sycl::half` (tandem-cuda: `fill_f16_bits`) | 1370 | 1364 |
| `fill` `uint8_t` | 1263 | 1330 |
| `fill` `bool` (one byte per bit) | 1250 | 1212 |
| `fill` `std::complex<double>` | 1395 | |
| `fill_below` `uint32_t`, range 1000 | 1378 | 1336 |
| `fill_below` `uint64_t`, range 1000 | 1392 | 1348 |
| `fill_below` `uint32_t`, range 1000, chunk kernel | 1339 | |
| `fill_normal` `float` | 1314 | 1290 |
| `fill_normal` `float`, start at an odd Float32 draw | 1330 | |
| `fill_normal` `float`, start at word 4 (first block at lane 1) | 1330 | |
| `fill_normal` `double` | 839 | 833 |
| `fill_normal` `double`, start at an odd Float64 draw | 771 | 679 |
| `fill_normal` `double`, start at word 6 | 782 | |

The uniform and bounded fills run at the card's memory bandwidth, as in tandem-cuda. The
normal fills reach the card's 250 W power cap, so they slow down as the card warms: a second
run right after this one gave 1297 for the float normal fill, 1133 for its odd start and 821
for the double one. Alternating the rows on a warm
card, the starts that shift the pairs within the blocks cost 8 % to 9 % against a start at a
multiple of 32 words.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
