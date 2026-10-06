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
on the same card by the same method. The cuRAND column is Philox4x32-10 of cuRAND 10.3.10 in the
same run, by the same method, through the CUDA runtime that AdaptiveCpp's CUDA backend uses. Where
cuRAND has no such output the column gives the nearest call, marked "nearest": it has no 8-, 16-
or 64-bit integers for Philox, so `curandGenerate` writes the same bytes as 32-bit words, and no
half floats, bools, bounded integers or exponentials. A normal row's start draw is a generator
offset in cuRAND. Narrow types write fewer bytes for the same element count.

| GiB/s written | tandem-sycl | tandem-cuda | cuRAND Philox4x32-10 | cuRAND call |
|---|---|---|---|---|
| `fill` `uint32_t`, tile kernel (default on GPUs for K >= 8) | 1361 | 1377 | 1255 | `curandGenerate` |
| `fill` `uint64_t`, tile kernel | 1381 | 1388 | 1295 | `curandGenerate`, nearest |
| `fill` `float`, tile kernel | 1359 | 1379 | 1245 | `curandGenerateUniform` |
| `fill` `double`, tile kernel | 1378 | 1388 | 778 | `curandGenerateUniformDouble` |
| `fill` `uint32_t`, chunk kernel | 1280 | 1311 | 1255 | `curandGenerate` |
| `fill` `uint64_t`, chunk kernel | 1295 | 1316 | 1295 | `curandGenerate`, nearest |
| `fill` `float`, chunk kernel | 1279 | 1306 | 1245 | `curandGenerateUniform` |
| `fill` `double`, chunk kernel | 1291 | 1315 | 778 | `curandGenerateUniformDouble` |
| `fill` `uint16_t` | 1325 | 1356 | 1222 | `curandGenerate`, nearest |
| `fill` `sycl::half` (tandem-cuda: `fill_f16_bits`) | 1236 | 1330 | 1222 | `curandGenerate`, nearest |
| `fill` `uint8_t` | 1125 | 1320 | 1155 | `curandGenerate`, nearest |
| `fill` `bool` (one byte per bit) | 1187 | 1209 | 1155 | `curandGenerate`, nearest |
| `fill` `std::complex<double>` | 1387 | | 788 | `curandGenerateUniformDouble` |
| `fill_below` `uint32_t`, range 1000 | 1359 | 1334 | 1255 | `curandGenerate`, nearest |
| `fill_below` `uint64_t`, range 1000 | 1382 | 1354 | 1295 | `curandGenerate`, nearest |
| `fill_below` `uint32_t`, range 1000, chunk kernel | 1317 | | 1255 | `curandGenerate`, nearest |
| `fill_normal` `float` | 1160 | 1172 | 858 | `curandGenerateNormal` |
| `fill_normal` `float`, start at an odd Float32 draw | 1063 | | 737 | `curandGenerateNormal`, offset 1 |
| `fill_normal` `float`, start at word 4 (first block at lane 1) | 1093 | | 806 | `curandGenerateNormal`, offset 4 |
| `fill_normal` `double` | 1066 | 1056 | 566 | `curandGenerateNormalDouble` |
| `fill_normal` `double`, start at an odd Float64 draw | 993 | 960 | 547 | `curandGenerateNormalDouble`, offset 1 |
| `fill_normal` `double`, start at word 6 | 996 | 922 | 548 | `curandGenerateNormalDouble`, offset 3 |
| `fill_exponential` `float` | 964 | 1150 | 1245 | `curandGenerateUniform`, nearest |
| `fill_exponential` `double` | 874 | 931 | 778 | `curandGenerateUniformDouble`, nearest |

The uniform and bounded fills run near the card's memory bandwidth, as in tandem-cuda, except the
8- and 16-bit fills, 2 % to 15 % below tandem-cuda's. The float
normal fills from an odd draw and from word 4 cost 6 % to 9 % against a start at a multiple of
32 words. The double normal rows are the ziggurat with tandem-cuda's two kernels. Their table
pass stores in octets, which keeps a start at word 6 at the rate of an odd start. When each lane
stored its own block, word 6 put every group's 128 bytes 16 bytes off the 128-byte lines,
touched five 32-byte sectors per group and step instead of four, and ran at 708 GiB/s by the
earlier method.

The normal and exponential fills run below the uniforms because the card holds 250 W: their
arithmetic lowers its clock until it bounds them, see tandem-cuda's
[design](https://github.com/tandem-rng/tandem-cuda/blob/main/docs/design.md). The exponentials
take tandem-cuda's folded logarithm, the same bits with two products fewer. tandem-cuda's device
f32 division is PTX, which AdaptiveCpp's generic target does not compile, so the f32 exponential
keeps the IEEE division and runs at 964 GiB/s against tandem-cuda's 1150.
