# Speed

`pixi run -e cuda bench-cuda` produces the figures.

## GPU

NVIDIA A100 40 GB (PCIe), driver 570, AdaptiveCpp 25.10.0 for CUDA 12.9 (`pixi run -e cuda
bench-cuda`): 2^28 elements into USM device memory, the minimum of 15 runs of eight
back-to-back fills after a half-second warm-up. The GPU had no other process for 30 seconds
before the run and none during it. The tandem-cuda column is the figure of its [speed page](https://github.com/tandem-rng/tandem-cuda/blob/main/docs/speed.md) on the same
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
| `fill_normal` `double` | 1106 | 1065 |
| `fill_normal` `double`, start at an odd Float64 draw | 1043 | 1102 |
| `fill_normal` `double`, start at word 6 | 708 | |
| `fill_exponential` `float` | 1099 | 1022 |
| `fill_exponential` `double` | 895 | 948 |

The uniform and bounded fills run at the card's memory bandwidth, as in tandem-cuda. The
normal and exponential fills reach the card's 250 W power cap, so they slow down as the card
warms: a second run right after this one gave 1297 for the float normal fill, 1133 for its odd
start, 821 for the double one and 975 for the float exponentials. Alternating the rows on a warm
card, the starts that shift the pairs within the blocks cost 8 % to 9 % against a start at a
multiple of 32 words.

The double normal rows are the ziggurat with tandem-cuda's two kernels, measured on 2026-10-05
with the same method. Two runs agreed within 3 %. A start at word 6 puts every group's 128
bytes 16 bytes off the card's 128-byte lines, so its stores touch five 32-byte sectors per group
and step instead of four.

The exponential rows compare against tandem-cuda only, because SYCL has no standard
exponential sampler.
