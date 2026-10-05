# Speed

`pixi run -e cuda bench-cuda` produces the figures.

## GPU

NVIDIA A100 40 GB (PCIe), driver 570, AdaptiveCpp 25.10.0 for CUDA 12.9 (`pixi run -e cuda
bench-cuda`): 2^28 elements into USM device memory, by tandem-cuda's method. Each row first runs
its own fill for two seconds, then times 21 fills, each to the queue's wait. `build/cuda/bench`
prints the median and the fastest of the 21, and the table gives the median. After the two
seconds every fill runs at the card's 250 W power cap, so the rows give the capped rate. A row
run alone (`build/cuda/bench <part of its label>`) gives the figure it gives in the table, within
3 %. Two runs agreed within 3 %. The GPU had no other process during the runs. The tandem-cuda
column is the figure of its [speed page](https://github.com/tandem-rng/tandem-cuda/blob/main/docs/speed.md)
on the same card by the same method. Narrow types write fewer bytes for the same element count.

| GiB/s written | tandem-sycl | tandem-cuda |
|---|---|---|
| `fill` `uint32_t`, tile kernel (default on GPUs for K >= 8) | 1359 | 1379 |
| `fill` `uint64_t`, tile kernel | 1380 | 1384 |
| `fill` `float`, tile kernel | 1359 | 1375 |
| `fill` `double`, tile kernel | 1377 | 1387 |
| `fill` `uint32_t`, chunk kernel | 1276 | 1300 |
| `fill` `uint64_t`, chunk kernel | 1297 | 1305 |
| `fill` `float`, chunk kernel | 1278 | 1304 |
| `fill` `double`, chunk kernel | 1293 | 1304 |
| `fill` `uint16_t` | 1320 | 1360 |
| `fill` `sycl::half` (tandem-cuda: `fill_f16_bits`) | 1247 | 1327 |
| `fill` `uint8_t` | 1129 | 1320 |
| `fill` `bool` (one byte per bit) | 1176 | 1233 |
| `fill` `std::complex<double>` | 1388 | |
| `fill_below` `uint32_t`, range 1000 | 1342 | 1334 |
| `fill_below` `uint64_t`, range 1000 | 1380 | 1352 |
| `fill_below` `uint32_t`, range 1000, chunk kernel | 1314 | |
| `fill_normal` `float` | 1187 | 1167 |
| `fill_normal` `float`, start at an odd Float32 draw | 1060 | |
| `fill_normal` `float`, start at word 4 (first block at lane 1) | 1092 | |
| `fill_normal` `double` | 1057 | 1060 |
| `fill_normal` `double`, start at an odd Float64 draw | 1005 | 972 |
| `fill_normal` `double`, start at word 6 | 1015 | 931, 954 |
| `fill_exponential` `float` | 955 | 1022 |
| `fill_exponential` `double` | 892 | 931 |

The uniform and bounded fills run near the card's memory bandwidth, as in tandem-cuda, except the
8- and 16-bit fills, 3 % to 15 % below tandem-cuda's. The float
normal fills from an odd draw and from word 4 cost 8 % to 11 % against a start at a multiple of
32 words. The double normal rows are the ziggurat with tandem-cuda's two kernels. Their table
pass stores in octets, which keeps a start at word 6 at the rate of an odd start. When each lane
stored its own block, word 6 put every group's 128 bytes 16 bytes off the 128-byte lines,
touched five 32-byte sectors per group and step instead of four, and ran at 708 GiB/s by the
earlier method.

The exponential rows compare against tandem-cuda only, because SYCL has no standard
exponential sampler.
