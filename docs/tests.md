# Tests

```sh
pixi run test                         # AdaptiveCpp 25.10, OpenMP backend on the host CPU
pixi run -e dpcpp test-dpcpp          # DPC++ 2026.1 with Intel's OpenCL CPU runtime, Linux
pixi run -e cuda test-cuda            # AdaptiveCpp 25.10 for CUDA 12.9, Linux with an NVIDIA GPU
```

## Suite

`tests/test_tandem.cpp` runs every check on the default SYCL device: 3366 checks on the
AdaptiveCpp CPU device, 3942 on the A100, which also runs the normal fills' shuffle
variant. It
checks every vector of the specification and every case of its conformance files (see
below), hashes fills with both kernels and in-kernel scalar draws against the stream SHA-256
sums of `hashes.json`, compares fills against
in-kernel draws at random keys, chunk lengths, positions, lengths and output alignments, checks
fills of every type (bool, 8 to 64-bit, float, double) against the bits of the u32 stream,
checks that fills split at arbitrary points with host draws between them continue one stream,
and checks mixed-width draws, random access and derived keys in kernels. In-kernel bounded
draws and double normals equal the host's, in-kernel float normals agree with them to the
tolerance in [design](design.md). Bounded
fills are checked against Appendix A written out on host generators, against the sequential
`urand(range)` calls. Double normal fills of both kernels must equal the scalar `normal()` calls bit for bit
at random keys, chunk lengths, positions, lengths and alignments, with misses and tail values
among them. A miss list too short for the fill must make the second kernel walk the fill and
give the same values. Fills from every first draw modulo 16, at K = 1 and 32, on and 8 bytes
off 16-byte addresses, must equal them too, with each way of passing values between lanes. A
fill cut at an odd element, at its first miss and just after must equal the
whole. Float normal
fills of every kernel variant are checked against the scalar `normalf2()` calls to 16 ulps plus
1e-6, from random positions and counts and from starts at every word offset within a block and
several lanes, with K from 1 to 64. Exponential
fills of both kernels must equal the scalar `exponential()` calls bit for bit, and 10^7
samples must match the Exp(1) law in their first four moments and a Kolmogorov-Smirnov test.
In-kernel exponentials equal the host's bit for bit. Signed, Float16, `sycl::half` and complex
fills, buffers of rank 1 and 2, and the position after empty fills at odd positions are checked
too.

## Conformance files

`tests/conformance/*.json` are byte-identical copies of tandem-spec f420545
`conformance/*.json`, read by `tests/conformance.hpp`. Each item of the spec's
`conformance/CHECKLIST.md` at b31af72, whose JSON files equal f420545's, has its test:

| checklist section | test |
|---|---|
| Fallback by global draw index | `check_fill_cases` on `fill_below.json` and `normal.json`, `check_shift` on the `_AT[4]`, `_AT[6]` and `CROSS_NORMAL[1]` pairs |
| Width from range | `check_width_from_range`: every u32 case of `fill_below.json` through `fill_below` into `uint64_t`, by USM and a buffer, `Rng::below` on `CROSS_BELOW32[3]` on the host and in a kernel, ranges 2^32 and 2^32 + 1; the u64 cases through their width-naming kind; `check_range0` |
| n = 0 | the seven `n = 0` cases through the public fills, onto a sentinel |
| Odd n | the `CROSS_NORMAL32` cases, values and end |
| Pair rule for Float32 Box-Muller | `CROSS_NORMALF`, `check_shift` on `CROSS_NORMAL32[1]` and `[2]`, `check_scalars` for `normalf()` |
| Weighted choice | `test_choice`: every table of `choice.json` (whole where pinned, else `capacity`), every case, `check_shift` on `CROSS_CHOICE[1]`, `check_scalars` for `choice()`, the `n = 0` case, rejected weights; `m = 1` is the `choice single` case |
| Cut fill | `check_fill_cases` cuts every case of the four fill files at 1, 7, 20, 21 and n − 1, Float32 normals at 2, 8, 20 and the largest even element below n; `check_scalars` |
| Block and 2^63 position boundaries | `test_streams`, `test_dumps`, `check_complex_straddle`, `check_random_access`, `check_position_bounds`, which runs every item: `Rng` holds any 64-bit position and a fill takes any 64-bit length. Uniform, bool, bounded, normal, exponential and choice fills that end exactly at 2^64 throw `std::length_error` and leave the memory and the position unchanged |

The fill cases run on every kernel path: chunk and tile, and for float normals the group
kernel with local memory and with shuffles. The Float32 normal dump hash
holds for C's polynomials, which the host's `normalf2()` takes, so that dump runs on the host.
`hashes.json` has no fill here for its UInt128 and Char streams. Choice fills of both kernels
also equal the `choice()` calls at random keys, chunk lengths, positions, lengths and output
alignments, and a rank 2 buffer equals the USM fill.

`tests/vectors.hpp` is generated from the spec repository's `vectors.json` by
`tools/gen_vectors.py`.

## CI

CI runs the tests on the CPU device with AdaptiveCpp on Linux and macOS and with DPC++ on
Linux, all with `-Wall -Wextra -Werror`, and fails when the vector header differs from
upstream, a conformance copy differs from tandem-spec f420545, or the submodule pin leaves
tandem-cuda's main. The suite also passes on
an NVIDIA A100 with the `cuda` environment.
