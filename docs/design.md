# Design

Bounded, normal and exponential draws follow Appendix A of the specification, the contract in
`core.hpp`, and the same fills in tandem-c, tandem-cuda and tandem-kokkos.

## Fills

One work item steps one chunk. On GPUs and other accelerators with `K >= 8` a work group of
256 items stages eight steps of 32 groups in 32 KiB of local memory and writes them as 512
contiguous bytes per 32 items. On CPU devices, and for smaller `K`, each item stores its blocks
directly. Both store whole 16-byte blocks when the output's blocks are 16-byte aligned, which a
kernel checks on the device, so USM and buffer outputs share one path. A bool block is 128
bytes and leaves as eight 16-byte stores. The exponential fills take the same kernels.

## Normals

Double normals are the 1024-layer ziggurat of Appendix A, one UInt64 draw per element, from
`core.hpp`, so `Rng::normal`, the fills, tandem-c and tandem-cuda agree bit for bit. Element
`i` of a fill takes draw `i`. A draw that misses the fast path, 0.43 % of them, continues on
`split(g)` of `sub(PURPOSE_NORMAL64)` of the key at position 0, where `g` is the draw's global
index, so a fill cut at any element equals the whole fill. The fill runs on the uniform fill's
kernels with two changes:

- A device puts `core.hpp`'s layer table in constant memory, whose reads serialize when the
  lanes of a sub-group read different layers. That cut the A100 to 27 GiB/s. So the kernels
  copy the 16 KiB of layers into local memory first.
- A miss continued in place runs its whole sub-group through the fallback for one lane. Even
  unexecuted, the call doubles the kernel's registers. So the GPU kernel stages four steps per
  pass instead of eight, writes the fast path's values, and queues the work group's misses in
  local memory. The work group continues them one per work item at its end, or after a pass
  that leaves the queue half full. A pass that overflows the queue continues its misses from
  the staged blocks. The CPU kernel and `K < 4` continue misses in place.

The float normal fill takes Box-Muller pairs with the logarithm of `core.hpp` and the fast
`sycl::native::cos` and `sin`, as tandem-cuda takes `__sincosf`, with the angle shifted by half
a turn into `[-pi, pi)`. It needs no double precision on the device. Float normals agree with
the other ports to 16 ulps plus 1e-6. Define `TANDEM_PRECISE_F32_NORMAL` for `box_muller2_f32`
from `core.hpp` instead.

On GPUs the float normal fill cuts the stream into units of four 32-bit words from the fill's
first word, two pairs, each one 16-byte store. Each work item steps only
its own chunk, and the eight lanes of a group write eight consecutive units, 128 aligned bytes,
one step after their rows are out. When the start is not a multiple of 32 words, a unit's
words lie in two blocks held by other lanes, which pass them by sub-group shuffles. The units
that end in the next group take its first row, kept from step 0, or for the group after the
work group computed in one step. So any start costs at most about a tenth more. The word
offset within a block is a template parameter, four kernels. Devices whose sub-group
sizes are not all multiples of eight pass the blocks through local memory with a barrier per
step instead. On CPU devices each item writes the pairs that end in its blocks, stepping the
previous chunk as well at such starts.

## Exponentials

The exponentials take the same polynomial logarithm in both precisions and equal tandem-c's bit
for bit on every tested device. Double fills, `Rng::normal` and `Rng::exponential` need a
device with `aspect::fp64`, `Rng::normalf` and `Rng::exponentialf` do not.
