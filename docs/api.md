# API

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
double *t = sycl::malloc_device<double>(n, q);
auto e3 = tandem::fill_exponential(q, t, n, rng);   // standard exponentials

const double w[] = {1, 2, 3, 4};
std::vector<uint64_t> cut(4);
std::vector<uint32_t> alias(4);
tandem::ChoiceTable table;
tandem::choice_build(table, w, 4, cut.data(), alias.data());   // on the host, no draws
uint64_t *dcut = sycl::malloc_device<uint64_t>(4, q);
uint32_t *dalias = sycl::malloc_device<uint32_t>(4, q);
q.copy(cut.data(), dcut, 4);
q.copy(alias.data(), dalias, 4).wait();
uint32_t *k = sycl::malloc_device<uint32_t>(n, q);
tandem::fill_choice(q, k, n, rng, {table.capacity, dcut, dalias, table.m});  // i with prob. w[i] / 10

float *y = sycl::malloc_device<float>(m, q);
q.parallel_for(sycl::range<1>(m), {e1, e2, e3}, [=](sycl::id<1> i) {
    tandem::Rng r = rng.split(i);           // one generator per work item, from the key alone
    y[i] = r.frand() + r.frand();
}).wait();
```

## Reference

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
- `tandem::fill_normal(queue, ptr, n, rng)`: standard normals in `float` or `double`. Double:
  the 1024-layer ziggurat, the sequence of `Rng::normal` calls, element `i` from UInt64 draw
  `i`, bit for bit equal to tandem-c. Float: Box-Muller, the flattened sequence of
  `Rng::normalf2` calls. Pair `j`, the elements `2j` and `2j + 1` with the cos half first, comes
  from the draws `2j` and `2j + 1` of the Float32 fill. An odd count drops the last sin half
  and still consumes both draws, 64 bits per pair.
- `tandem::fill_exponential(queue, ptr, n, rng)`: standard exponentials `-log(1 - u)` in
  `float` or `double`, the sequence of `Rng::exponentialf` or `Rng::exponential` calls. Element
  `i` comes from draw `i` of the Float32 (Float64) fill.
- `tandem::fill_choice(queue, ptr, n, rng, table)` and `tandem::fill_choice(queue, buffer, rng,
  table)`: weighted choice indices in `uint32_t`, Appendix C of the specification. Element `i`
  maps UInt64 draw `i` through the alias table, so the fill consumes 64 bits per element,
  never retries, and equals the `Rng::choice` calls and tandem-c bit for bit. An empty fill
  aligns the position to 64 bits. `tandem::choice_build(table, weights, m, cut, alias)` from
  `core.hpp` builds the table on the host from `m` Float64 weights in exact integers. It
  returns false for no weights, a negative, infinite or NaN weight, or all zeros. The table
  views the `cut` and `alias` arrays: the fill needs device copies, host draws the host ones.
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
`fill` or double `fill_normal` aligns the position to the type's width, and an empty
`fill_below`, float `fill_normal` or `fill_exponential` consumes no draws and leaves the
position alone, and an empty `fill_choice` aligns it to 64 bits. A fill that would run past
stream position 2^64 throws `std::overflow_error` before it moves the position.

## Rng draws

| | |
|---|---|
| `bit()`, `urand()`, `urand64()`, `frand()`, `drand()` | the specification's Bool, UInt32, UInt64, Float32 and Float64 draws |
| `at_urand(i)`, `at_urand64(i)`, `at_frand(i)`, `at_drand(i)` | element `i` of the fill that would start here, without advancing |
| `urand(range)`, `urand64(range)`, `rand(start, end)`, `rand64(start, end)`, `frand(range)`, `drand(start, end)`, ... | bounded draws, uniform by Lemire's multiply and reject |
| `normal()`, `normal(mean, sd)` | the ziggurat of one UInt64 draw |
| `normalf()` | the cos half of a Box-Muller step from two Float32 draws |
| `normal2()`, `normalf2()` | two ziggurat draws, or both halves of the float step, as a pair `z0`, `z1` |
| `exponential()`, `exponentialf()` | `-log(1 - u)` of one Float64 or Float32 draw |
| `choice(table)` | the weighted choice of one UInt64 draw, element 0 of a choice fill |
| `split(i)`, `sub(purpose)`, `fork(children, n)` | child generators as the specification defines them |
| `key()`, `position()`, `set_position(p)`, `chunk_length()` | transport form |

Signed integers hold the two's complement of the unsigned draw of the same width. A complex
value takes two draws, the real and then the imaginary component.
