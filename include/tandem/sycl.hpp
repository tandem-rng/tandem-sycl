/* Tandem8x32 for SYCL: fills of USM memory and buffers on any SYCL device, and the scalar
 * generator tandem::Rng from tandem/core.hpp for draws inside kernels. Header only, C++20.
 *
 * Implements https://github.com/tandem-rng/spec and produces the stream it defines, bit for
 * bit. Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
 */
#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <bit>
#include <complex>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <tandem/core.hpp>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "tandem/sycl.hpp assumes a little-endian host and device"
#endif

namespace tandem {

namespace detail {

struct f16_bits {}; /* binary16 bit patterns of the Float16 draws, stored as uint16_t */
struct below32 {};  /* Lemire bounded draws over the u32 fill, see PURPOSE_BELOW32 */
struct below64 {};
struct exp32 {}; /* -log(1 - u) of the f32 draws, spec Appendix A */
struct exp64 {};
struct normal64 {}; /* the ziggurat of the u64 draws, see PURPOSE_NORMAL64 */
struct choice_idx {}; /* weighted choice indices of the u64 draws, spec Appendix C */

/* The fill's geometry: stream bits [p0, p1), rows r0 .. r1 inclusive, groups g0 .. g1, and
 * what the bounded kinds, the f64 normals and the choice need: the fill's key and chunk length,
 * the range and its rejection threshold, the ziggurat's layers, which a kernel points to its
 * copy in local memory, and the alias table. The float normal fill keeps its element count in
 * `range`. */
struct Span {
    uint64_t p0, p1, r0, r1, g0, g1;
    uint32_t K;
    Key key;
    uint64_t range, thresh;
    const zig::Layer *wk;
    ChoiceTable table;
};

/* How an output element is made from a block: element k of the block takes bits
 * [k bits, (k + 1) bits). The bounded kinds give the draw, which block_values maps. */
template <class Kind> struct elem;

template <> struct elem<bool> {
    using out_t = bool;
    static constexpr unsigned bits = 1; /* see store_bool_part */
};
template <> struct elem<uint8_t> {
    using out_t = uint8_t;
    static constexpr unsigned bits = 8;
    static uint8_t make(const uint32_t w[4], unsigned k) {
        return (uint8_t)(w[k >> 2] >> ((k & 3u) * 8u));
    }
};
template <> struct elem<uint16_t> {
    using out_t = uint16_t;
    static constexpr unsigned bits = 16;
    static uint16_t make(const uint32_t w[4], unsigned k) {
        return (uint16_t)(w[k >> 1] >> ((k & 1u) * 16u));
    }
};
template <> struct elem<f16_bits> {
    using out_t = uint16_t;
    static constexpr unsigned bits = 16;
    static uint16_t make(const uint32_t w[4], unsigned k) {
        return to_f16_bits(elem<uint16_t>::make(w, k));
    }
};
template <> struct elem<uint32_t> {
    using out_t = uint32_t;
    static constexpr unsigned bits = 32;
    static uint32_t make(const uint32_t w[4], unsigned k) { return w[k]; }
};
template <> struct elem<float> {
    using out_t = float;
    static constexpr unsigned bits = 32;
    static float make(const uint32_t w[4], unsigned k) {
        return to_f32(w[k]);
    }
};
template <> struct elem<uint64_t> {
    using out_t = uint64_t;
    static constexpr unsigned bits = 64;
    static uint64_t make(const uint32_t w[4], unsigned k) {
        return w[2 * k] | ((uint64_t)w[2 * k + 1] << 32);
    }
};
template <> struct elem<double> {
    using out_t = double;
    static constexpr unsigned bits = 64;
    static double make(const uint32_t w[4], unsigned k) {
        return to_f64(w[2 * k] | ((uint64_t)w[2 * k + 1] << 32));
    }
};
template <> struct elem<exp32> {
    using out_t = float;
    static constexpr unsigned bits = 32;
    static float make(const uint32_t w[4], unsigned k) { return exponential_f32(to_f32(w[k])); }
};
template <> struct elem<exp64> {
    using out_t = double;
    static constexpr unsigned bits = 64;
    static double make(const uint32_t w[4], unsigned k) {
        return exponential_f64(elem<double>::make(w, k));
    }
};
template <> struct elem<normal64> {
    using out_t = double;
    static constexpr unsigned bits = 64;
    static uint64_t draw(const uint32_t w[4], unsigned k) {
        return w[2 * k] | ((uint64_t)w[2 * k + 1] << 32);
    }
};
template <> struct elem<choice_idx> {
    using out_t = uint32_t;
    static constexpr unsigned bits = 64;
    static uint64_t draw(const uint32_t w[4], unsigned k) {
        return w[2 * k] | ((uint64_t)w[2 * k + 1] << 32);
    }
};
template <> struct elem<below32> {
    using out_t = uint32_t;
    static constexpr unsigned bits = 32;
    static uint32_t draw(const uint32_t w[4], unsigned k) { return w[k]; }
};
template <> struct elem<below64> {
    using out_t = uint64_t;
    static constexpr unsigned bits = 64;
    static uint64_t draw(const uint32_t w[4], unsigned k) {
        return w[2 * k] | ((uint64_t)w[2 * k + 1] << 32);
    }
};

struct alignas(16) Block {
    uint32_t w[4];
};

/* The output bytes of one block: 16, or 8 for the choice's u32 indices of u64 draws. A bool
 * block, 128 bytes, leaves in 16-byte parts. */
template <class Kind>
constexpr unsigned block_bytes =
    std::is_same_v<Kind, bool> ? 16 : 128 / elem<Kind>::bits * sizeof(typename elem<Kind>::out_t);

/* Whether every block of the fill lands on an address aligned to its output bytes: the output's
 * first byte and the output offset of the fill's first stream bit agree modulo block_bytes. */
template <class Kind> bool blocks_aligned(const typename elem<Kind>::out_t *out, const Span &s) {
    uint64_t first = s.p0 / elem<Kind>::bits * sizeof(typename elem<Kind>::out_t);
    return ((reinterpret_cast<uintptr_t>(out) - first) & (block_bytes<Kind> - 1u)) == 0;
}

/* An N-byte store, N = 8 or 16, to an N-byte aligned address. NVPTX receives a copy of a
 * 16-byte struct as narrower stores, so clang stores a vector type. */
template <unsigned N> inline void store_wide(void *dst, const void *src) {
#if defined(__clang__)
    typedef uint32_t words __attribute__((ext_vector_type(N / 4)));
    words x;
    std::memcpy(&x, src, N);
    *static_cast<words *>(dst) = x;
#else
    std::memcpy(__builtin_assume_aligned(dst, N), src, N);
#endif
}

/* Stream bits [P + 16 q, P + 16 q + 16) of a bool fill, which are 16 output bytes, from the
 * block w at stream bit P. */
template <bool ALIGNED>
inline void store_bool_part(bool *out, const Span &s, uint64_t P, const uint32_t w[4],
                            unsigned q) {
    uint64_t first = P + 16u * q;
    uint32_t x = w[q >> 1] >> ((q & 1u) * 16u);
    if (ALIGNED && first >= s.p0 && first + 16u <= s.p1) {
        Block v;
        /* Spread four bits over the low bits of four bytes. */
        for (unsigned t = 0; t < 4; t++)
            v.w[t] = (((x >> (4 * t)) & 0xfu) * 0x00204081u) & 0x01010101u;
        store_wide<16>(out + (first - s.p0), &v);
        return;
    }
    for (unsigned k = 0; k < 16; k++)
        if (first + k >= s.p0 && first + k < s.p1)
            out[first + k - s.p0] = (x >> k) & 1u;
}

template <class Kind>
constexpr bool is_below = std::is_same_v<Kind, below32> || std::is_same_v<Kind, below64>;

/* normal_f64_fast of core.hpp on the layers wk. */
inline double normal64_fast(const zig::Layer *wk, uint64_t r, bool &hit) {
    uint64_t ra = r >> 11;
    zig::Layer l = wk[r & 1023u];
    hit = ra < l.k;
    return std::bit_cast<double>(std::bit_cast<uint64_t>((double)ra * l.w) ^ ((r & 1024u) << 53));
}

/* The elements of the block at stream bit P, by Lemire's method for the bounded kinds, as
 * below_u32_t and below_u64_t in core.hpp, and by the ziggurat for normal64. Both accept or
 * flag every draw of the block first and then retry the flagged ones in one loop, because a
 * device compiler that inlines everything would otherwise copy the retry, a generator's
 * seeding, once per element. The retry of draw k takes the global draw index
 * (P + k bits) / bits, as Appendix A of the specification says, so a fill cut into pieces
 * equals the whole fill. */
template <class Kind>
inline void block_values(const uint32_t w[4], uint64_t P, const Span &s,
                         typename elem<Kind>::out_t x[]) {
    using O = typename elem<Kind>::out_t;
    constexpr unsigned bits = elem<Kind>::bits, per_block = 128 / bits;
    constexpr bool retries = is_below<Kind> || std::is_same_v<Kind, normal64>;
    if constexpr (retries) {
        unsigned rejected = 0;
        for (unsigned k = 0; k < per_block; k++) {
            if constexpr (std::is_same_v<Kind, normal64>) {
                bool hit;
                x[k] = normal64_fast(s.wk, elem<Kind>::draw(w, k), hit);
                rejected |= (unsigned)!hit << k;
            } else {
                const O range = (O)s.range;
                O u = elem<Kind>::draw(w, k), lo;
                if constexpr (sizeof(O) == 4) {
                    uint64_t m = (uint64_t)u * range;
                    x[k] = (O)(m >> 32);
                    lo = (O)m;
                } else {
                    x[k] = mulhi64(u, range);
                    lo = u * range;
                }
                if (lo < (O)s.thresh)
                    rejected |= 1u << k;
            }
        }
        while (rejected) {
            unsigned k = (unsigned)std::countr_zero(rejected);
            rejected &= rejected - 1u;
            uint64_t g = P / bits + k;
            O v;
            if constexpr (std::is_same_v<Kind, normal64>)
                v = normal_f64_slow(elem<Kind>::draw(w, k), s.key.w, s.K, g);
            else if constexpr (sizeof(O) == 4)
                v = below_retry_u32((O)s.range, (O)s.thresh, s.key.w, s.K, g);
            else
                v = below_retry_u64((O)s.range, (O)s.thresh, s.key.w, s.K, g);
            /* A select per element, since a variable index would put x in local memory. */
            for (unsigned j = 0; j < per_block; j++)
                if (j == k)
                    x[j] = v;
        }
    } else if constexpr (std::is_same_v<Kind, choice_idx>) {
        for (unsigned k = 0; k < per_block; k++)
            x[k] = choice_of(s.table, elem<Kind>::draw(w, k));
    } else {
        for (unsigned k = 0; k < per_block; k++)
            x[k] = elem<Kind>::make(w, k);
    }
}

/* Store the elements of the block at stream bit P that fall inside the fill's bits [p0, p1).
 * With ALIGNED a block fully inside leaves as one store of block_bytes, or eight 16-byte stores
 * for bool, whose 128 elements are one byte each. */
template <class Kind, bool ALIGNED>
inline void store_block(typename elem<Kind>::out_t *out, const Span &s, uint64_t P,
                        const uint32_t w[4]) {
    using O = typename elem<Kind>::out_t;
    constexpr unsigned bits = elem<Kind>::bits, per_block = 128 / bits;
    if constexpr (std::is_same_v<Kind, bool>) {
        for (unsigned q = 0; q < 8; q++)
            store_bool_part<ALIGNED>(out, s, P, w, q);
    } else {
        O x[per_block];
        block_values<Kind>(w, P, s, x);
        if (ALIGNED && P >= s.p0 && P + 128u <= s.p1) {
            store_wide<block_bytes<Kind>>(out + (P - s.p0) / bits, x);
            return;
        }
        for (unsigned k = 0; k < per_block; k++) {
            uint64_t q = P + k * bits;
            if (q >= s.p0 && q < s.p1)
                out[(q - s.p0) / bits] = x[k];
        }
    }
}

/* Auto is Tile on GPUs and other accelerators, else Chunk. The uniform and bounded fills take
 * Chunk for K < 8, the double normal fills below 2^16 elements. */
enum class Kernel {
    Auto,
    Chunk, /* one work item per chunk, one block per step */
    Tile,  /* one work group per 32 groups: eight steps staged in local memory, for float normal
            * fills the units of each row passed between lanes, and for double normal fills the
            * table pass and the miss kernel */
};

/* The output of a fill as the kernel sees it. A USM pointer is used as is. A buffer is bound
 * to the command group by an accessor, whose pointer is known only on the device, so kernels
 * check the alignment there. */
template <class O> struct UsmOut {
    O *p;
    O *get() const { return p; }
};

template <class O, class T, int D> struct AccOut {
    sycl::accessor<T, D, sycl::access_mode::write> acc;
    O *get() const {
        return reinterpret_cast<O *>(
            acc.template get_multi_ptr<sycl::access::decorated::no>().get());
    }
};

template <class O, class T, int D> struct BufOut {
    sycl::buffer<T, D> &buf;
};

/* `keep` binds a buffer that an earlier kernel of the same fill has written. */
template <class O> UsmOut<O> bind(const UsmOut<O> &u, sycl::handler &, bool = false) { return u; }
template <class O, class T, int D>
AccOut<O, T, D> bind(const BufOut<O, T, D> &b, sycl::handler &h, bool keep = false) {
    if (keep)
        return AccOut<O, T, D>{sycl::accessor(b.buf, h, sycl::write_only)};
    return AccOut<O, T, D>{sycl::accessor(b.buf, h, sycl::write_only, sycl::no_init)};
}

template <class Kind, bool ALIGNED>
inline void chunk_body(typename elem<Kind>::out_t *out, const Span &s, uint64_t t) {
    uint64_t c = 8u * s.g0 + t, row = (c >> 3) * s.K;
    uint32_t o[4], h[4];
    F_keyed(s.key.w, c, DOMAIN_STREAM, AUX_STREAM, o, h);
    uint32_t j0 = row < s.r0 ? (uint32_t)(s.r0 - row) : 0u;
    uint32_t j1 = (uint32_t)(s.r1 - row < s.K - 1u ? s.r1 - row : s.K - 1u);
    for (uint32_t j = 0; j <= j1; j++) {
        T(o, h);
        if (j >= j0)
            store_block<Kind, ALIGNED>(out, s, (row + j) * 1024u + (c & 7u) * 128u, o);
    }
}

constexpr unsigned TILE_ITEMS = 256, TILE_GROUPS = TILE_ITEMS / 8, TILE_STEPS = 8;
constexpr unsigned TILE_SLOTS = TILE_GROUPS * TILE_STEPS * 8;

/* The ziggurat's layers in the work group's local memory, for the f64 normals. A device puts
 * zig::WK in constant memory, whose reads serialize when the lanes read different layers, which
 * cuts the fill to a fiftieth of its speed on an A100. */
inline Span with_layers(const Span &s, sycl::nd_item<1> it, zig::Layer *wk) {
    for (size_t i = it.get_local_id(0); i < 1024; i += it.get_local_range(0))
        wk[i] = zig::WK[i];
    sycl::group_barrier(it.get_group());
    Span t = s;
    t.wk = wk;
    return t;
}

template <class Kind, class Out>
sycl::event fill_chunk(sycl::queue &q, const Span s, const Out &out,
                       const std::vector<sycl::event> &deps) {
    const uint64_t items = 8u * (s.g1 - s.g0 + 1u);
    return q.submit([&](sycl::handler &h) {
        h.depends_on(deps);
        auto dst = bind(out, h);
        if constexpr (std::is_same_v<Kind, normal64>) {
            sycl::local_accessor<zig::Layer, 1> wk(sycl::range<1>(1024), h);
            uint64_t teams = (items + TILE_ITEMS - 1u) / TILE_ITEMS;
            h.parallel_for(sycl::nd_range<1>(teams * TILE_ITEMS, TILE_ITEMS),
                           [=](sycl::nd_item<1> it) {
                               auto *p = dst.get();
                               const Span t = with_layers(
                                   s, it,
                                   wk.template get_multi_ptr<sycl::access::decorated::no>().get());
                               uint64_t c = it.get_global_id(0);
                               if (c >= items)
                                   return;
                               if (blocks_aligned<Kind>(p, t))
                                   chunk_body<Kind, true>(p, t, c);
                               else
                                   chunk_body<Kind, false>(p, t, c);
                           });
        } else {
            h.parallel_for(sycl::range<1>(items), [=](sycl::item<1> it) {
                auto *p = dst.get();
                if (blocks_aligned<Kind>(p, s))
                    chunk_body<Kind, true>(p, s, it.get_id(0));
                else
                    chunk_body<Kind, false>(p, s, it.get_id(0));
            });
        }
    });
}

/* One work item per chunk, 32 groups per work group. Every TILE_STEPS steps the work group
 * holds, for each of its groups, TILE_STEPS consecutive rows, 1024 contiguous bytes of the
 * stream. The write phase hands consecutive 16-byte slots to consecutive work items, so a
 * 32-wide sub-group writes 512 contiguous bytes. Needs K >= TILE_STEPS. */
template <class Kind, bool ALIGNED>
inline void tile_body(typename elem<Kind>::out_t *out, const Span &s, sycl::nd_item<1> it,
                      Block *tile) {
    uint64_t gb = s.g0 + (uint64_t)it.get_group(0) * TILE_GROUPS;
    unsigned rank = (unsigned)it.get_local_id(0), gi = rank >> 3, lane = rank & 7u;
    bool mine = gb + gi <= s.g1;
    uint32_t o[4], h[4];
    F_keyed(s.key.w, 8u * (gb + gi) + lane, DOMAIN_STREAM, AUX_STREAM, o, h);
    uint64_t first_bit = gb * s.K * 1024u;
    for (uint32_t jb = 0; jb < s.K; jb += TILE_STEPS) {
        if (first_bit + jb * 1024u >= s.p1)
            break;
        if (mine) {
            for (unsigned j = 0; j < TILE_STEPS; j++) {
                T(o, h);
                tile[(gi * TILE_STEPS + j) * 8 + lane] = Block{{o[0], o[1], o[2], o[3]}};
            }
        }
        sycl::group_barrier(it.get_group());
        /* A bool block is 128 output bytes, so consecutive work items take its 16-byte parts
         * to keep the stores contiguous. */
        constexpr unsigned parts = std::is_same_v<Kind, bool> ? 8 : 1;
        for (unsigned u = rank; u < TILE_SLOTS * parts; u += TILE_ITEMS) {
            unsigned slot = u / parts;
            unsigned sg = slot / (TILE_STEPS * 8), within = slot % (TILE_STEPS * 8);
            uint64_t P = ((gb + sg) * s.K + jb) * 1024u + within * 128u;
            if (P >= s.p1)
                continue;
            if constexpr (std::is_same_v<Kind, bool>)
                store_bool_part<ALIGNED>(out, s, P, tile[slot].w, u % parts);
            else
                store_block<Kind, ALIGNED>(out, s, P, tile[slot].w);
        }
        sycl::group_barrier(it.get_group());
    }
}

template <class Kind, class Out>
sycl::event fill_tile(sycl::queue &q, const Span s, const Out &out,
                      const std::vector<sycl::event> &deps) {
    uint64_t teams = (s.g1 - s.g0 + TILE_GROUPS) / TILE_GROUPS;
    return q.submit([&](sycl::handler &h) {
        h.depends_on(deps);
        auto dst = bind(out, h);
        sycl::local_accessor<Block, 1> tile(sycl::range<1>(TILE_SLOTS), h);
        h.parallel_for(sycl::nd_range<1>(teams * TILE_ITEMS, TILE_ITEMS),
                       [=](sycl::nd_item<1> it) {
                           auto *p = dst.get();
                           Block *t = tile.template get_multi_ptr<sycl::access::decorated::no>()
                                          .get();
                           if (blocks_aligned<Kind>(p, s))
                               tile_body<Kind, true>(p, s, it, t);
                           else
                               tile_body<Kind, false>(p, s, it, t);
                       });
    });
}

/* Whether every sub-group size of the device is a multiple of eight, so that a sub-group holds
 * whole groups of eight chunks. */
inline bool subgroups_hold_groups(const sycl::device &dev) {
    auto sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    for (size_t z : sizes)
        if (z % 8u != 0)
            return false;
    return !sizes.empty();
}

/* The f64 normal fill on GPUs, the two kernels of tandem-cuda. The table pass writes the fast
 * path's values and queues the work group's misses, 0.43 % of the elements, in local memory. At
 * its end the work group appends the queue to a list in device memory with one atomic add, and a
 * miss past the queue goes to the list directly. The second kernel continues each listed miss,
 * one work item per miss, so the table pass makes no calls and keeps its registers. When the
 * list overflows, the second kernel walks the whole fill again and continues every miss.
 *
 * The list and a copy of the layers live in device memory kept per context and device, and the
 * fills that share it chain on its last event. A device puts zig::WK in constant memory, whose
 * reads serialize when the lanes read different layers. */
constexpr unsigned NORMAL_QUEUE = 256; /* per work group, four times the mean at K = 32 */
constexpr uint64_t NORMAL_LIST_MIN = (uint64_t)1 << 16; /* shorter fills take the chunk kernel */

struct NormalMiss {
    uint64_t e, r; /* element index, draw */
};

/* The two fast-path values of one block. */
struct Pair64 {
    double v[2];
};

/* How a kernel passes values between the lanes of a group. Auto takes shuffles on devices whose
 * sub-group sizes are all multiples of eight, so that a sub-group holds whole groups. */
enum class Exchange { Auto, Shuffle, Local };

/* The local memory of normal64_octet_body: the miss queue, for a shifted start the first row of
 * each group and of the group after the work group, and without shuffles two steps of the
 * blocks passed between lanes. */
template <bool SHIFTED, bool SHUFFLE> struct NormalLocal {
    NormalMiss queue[NORMAL_QUEUE];
    Pair64 first[SHIFTED ? TILE_ITEMS + 8u : 1u];
    Pair64 xbuf[SHIFTED && !SHUFFLE ? 2u * TILE_ITEMS : 1u];
    uint64_t base;
    uint32_t queued;
};

/* The table pass, one work item per chunk as in tandem-cuda, with the stores of the float normal
 * fill's octets (normal_group_body). Each work item takes the fast path of its own block's two
 * draws and queues their misses. The fill is cut into units of two elements from its first draw
 * D0: unit u is 16 output bytes at element 2u and starts at draw KD = D0 % 2 of block B0 + u,
 * B0 = D0 / 2. Lane l of a group writes unit 8m + l of octet m, so the eight lanes write 128
 * aligned bytes whatever the start. With d = B0 % 8 the octet's units start in row R from lane d
 * on and in row R + 1 below lane d, so a group writes the octet of row R one step later, when row
 * R + 1 is out, and passes the values between lanes as the float fill passes words. A start
 * 16 bytes off the lines cost the stores a fifth sector per group and step, a third of the
 * speed. KD and SHIFTED (KD != 0 or d != 0) are template parameters, as in the float fill. */
template <unsigned KD, bool SHIFTED, bool SHUFFLE>
inline void normal64_octet_body(double *out, const Span &s, sycl::nd_item<1> it,
                                NormalLocal<SHIFTED, SHUFFLE> &lm, NormalMiss *list,
                                unsigned long long *count, uint64_t cap) {
    const unsigned rank = (unsigned)it.get_local_id(0), gi = rank >> 3, lane = rank & 7u;
    const uint64_t gb = s.g0 + (uint64_t)it.get_group(0) * TILE_GROUPS, g = gb + gi;
    const bool mine = g <= s.g1;
    const uint64_t D0 = s.p0 >> 6, n = (s.p1 - s.p0) >> 6, units = (n + 1u) / 2u;
    const uint64_t B0 = D0 >> 1, R0 = B0 >> 3; /* octet m lies at rows R0 + m and R0 + m + 1 */
    const unsigned d = (unsigned)(B0 & 7u);
    const bool wide = (reinterpret_cast<uintptr_t>(out) & 15u) == 0;
    /* The lanes of this lane's unit's first and second block, and whether each lies in R + 1. */
    const unsigned la = (d + lane) & 7u, lb = (d + lane + 1u) & 7u;
    const bool na = d + lane >= 8u, nb = d + lane + 1u >= 8u;
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>
        queued(lm.queued);
    sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        listed(*count);
    if (rank == 0)
        lm.queued = 0;
    uint32_t o[4], h[4];
    F_keyed(s.key.w, 8u * g + lane, DOMAIN_STREAM, AUX_STREAM, o, h);
    if (SHIFTED && rank < 8u) {
        uint32_t w[4];
        block(s.key.w, 8u * (gb + TILE_GROUPS) + rank, 0, w);
        bool hit;
        for (unsigned k = 0; k < 2; k++)
            lm.first[TILE_ITEMS + rank].v[k] = normal64_fast(s.wk, elem<normal64>::draw(w, k), hit);
    }
    sycl::group_barrier(it.get_group());

    /* Step j of this lane's chunk: the fast path of its block, its misses queued. */
    auto step = [&](uint32_t j, Pair64 &x) {
        if (!mine)
            return;
        T(o, h);
        const uint64_t P = (g * s.K + j) * 1024u + lane * 128u;
        for (unsigned k = 0; k < 2; k++) {
            const uint64_t r = elem<normal64>::draw(o, k), q = P + 64u * k;
            bool hit;
            x.v[k] = normal64_fast(s.wk, r, hit);
            if (!hit && q >= s.p0 && q < s.p1) {
                NormalMiss m{(q - s.p0) / 64u, r};
                uint32_t at = queued.fetch_add(1u);
                if (at < NORMAL_QUEUE) {
                    lm.queue[at] = m;
                } else {
                    uint64_t t = listed.fetch_add(1ull);
                    if (t < cap)
                        list[t] = m;
                }
            }
        }
    };
    /* Unit 8 (R - R0) + lane: element t is value (t + KD) % 2 of its first block (t + KD < 2)
     * or of its second, taken from row R + 1 (late) or row R (early). */
    auto emit = [&](uint64_t R, const Pair64 &late, const Pair64 &early) {
        if (R < R0 || 8u * (R - R0) + lane >= units)
            return;
        const uint64_t e = 2u * (8u * (R - R0) + lane);
        double z[2];
        for (unsigned t = 0; t < 2; t++) {
            const unsigned k = (t + KD) & 1u;
            z[t] = (t + KD < 2u ? na : nb) ? late.v[k] : early.v[k];
        }
        if (wide && e + 2u <= n) {
            store_wide<16>(out + e, z);
        } else {
            for (unsigned t = 0; t < 2; t++)
                if (e + t < n)
                    out[e + t] = z[t];
        }
    };
    /* Value k of the block of lane la (k >= KD) or lb (k < KD). Every work item calls it. */
    auto fetch = [&](unsigned buf, const Pair64 &own, Pair64 &x) {
        if constexpr (SHUFFLE) {
            (void)buf;
            auto sg = it.get_sub_group();
            unsigned base = (unsigned)sg.get_local_linear_id() - lane;
            for (unsigned k = 0; k < 2; k++)
                x.v[k] = sycl::select_from_group(sg, own.v[k], base + (k >= KD ? la : lb));
        } else if constexpr (SHIFTED) {
            Pair64 *xa = lm.xbuf + buf * TILE_ITEMS;
            xa[rank] = own;
            sycl::group_barrier(it.get_group());
            for (unsigned k = 0; k < 2; k++)
                x.v[k] = xa[rank - lane + (k >= KD ? la : lb)].v[k];
        }
    };

    /* Whether step j has a row in the fill, the same for the whole work group. */
    auto live = [&](uint32_t j) { return j < s.K && gb * s.K + j <= s.r1; };
    Pair64 own{{0.0, 0.0}};
    if constexpr (!SHIFTED) {
        /* Units are the blocks themselves. */
        for (uint32_t j = 0; live(j); j++) {
            step(j, own);
            if (mine)
                emit(g * s.K + j, own, own);
        }
    } else {
        /* Step 0 is peeled so that the loop holds no test for it. */
        uint32_t j = 0;
        Pair64 prev{{0.0, 0.0}};
        if (live(0)) {
            step(0, own);
            lm.first[gi * 8u + lane] = own;
            fetch(0, own, prev);
            for (j = 1; live(j); j++) {
                step(j, own);
                Pair64 cur;
                fetch(j & 1u, own, cur);
                if (mine)
                    emit(g * s.K + j - 1u, cur, prev);
                prev = cur;
            }
        }
        /* The octet of the last row stepped. Its row R + 1 is the next group's first row after
         * a whole group, and after a loop cut short lies past the fill, so the units that need
         * it are past the fill too. */
        sycl::group_barrier(it.get_group());
        const Pair64 *next = lm.first + (gi + 1u) * 8u;
        Pair64 cur;
        for (unsigned k = 0; k < 2; k++)
            cur.v[k] = next[k >= KD ? la : lb].v[k];
        if (mine && j > 0)
            emit(g * s.K + j - 1u, cur, prev);
    }
    sycl::group_barrier(it.get_group());
    const uint32_t m = sycl::min(lm.queued, NORMAL_QUEUE);
    if (rank == 0)
        lm.base = m ? listed.fetch_add((unsigned long long)m) : 0;
    sycl::group_barrier(it.get_group());
    for (uint32_t i = rank; i < m; i += TILE_ITEMS)
        if (lm.base + i < cap)
            list[lm.base + i] = lm.queue[i];
}

/* normal_f64_slow of core.hpp, operation for operation, from the fill's sub(PURPOSE_NORMAL64)
 * key, which the host derives once instead of every miss again, and the layers of wk. */
inline double normal64_continue(uint64_t r, const Key &sub, uint32_t K, uint64_t g,
                                const zig::Layer *wk) {
    using namespace detail;
    Rng fb = Rng::from_key(sub, 0, K).split(g);
    bool hit;
    double x = normal64_fast(wk, r, hit);
    for (;;) {
        unsigned i = (unsigned)r & 1023u;
        if (i == 0) {
            double a, b;
            do {
                a = exponential_f64(fb.drand()) / zig::R;
                b = exponential_f64(fb.drand());
            } while (b + b < a * a);
            double t = zig::R + a;
            return (r >> 10) & 1u ? -t : t;
        }
        double y = add_rn(zig_y(i), fmad(fb.drand(), zig_y(i + 1) - zig_y(i), 0.0));
        if (-neg_log_f64(y) < -0.5 * (x * x))
            return x;
        r = fb.urand64();
        x = normal64_fast(wk, r, hit);
        if (hit)
            return x;
    }
}

/* The listed misses from work item t of `items`, or all misses of the fill by chunk when the
 * list overflowed. */
inline void normal64_miss_body(double *out, const Span &s, const Key &sub, uint64_t t,
                               uint64_t items, const NormalMiss *list,
                               const unsigned long long *count, uint64_t cap) {
    const uint64_t listed = *count;
    if (listed <= cap) {
        for (uint64_t i = t; i < listed; i += items)
            out[list[i].e] = normal64_continue(list[i].r, sub, s.K, s.p0 / 64u + list[i].e, s.wk);
        return;
    }
    for (uint64_t c = 8u * s.g0 + t; c < 8u * (s.g1 + 1u); c += items) {
        uint64_t row = (c >> 3) * s.K;
        uint32_t o[4], h[4];
        F_keyed(s.key.w, c, DOMAIN_STREAM, AUX_STREAM, o, h);
        uint32_t j0 = row < s.r0 ? (uint32_t)(s.r0 - row) : 0u;
        uint32_t j1 = (uint32_t)(s.r1 - row < s.K - 1u ? s.r1 - row : s.K - 1u);
        for (uint32_t j = 0; j <= j1; j++) {
            T(o, h);
            for (unsigned k = 0; j >= j0 && k < 2; k++) {
                uint64_t q = (row + j) * 1024u + (c & 7u) * 128u + 64u * k;
                uint64_t r = elem<normal64>::draw(o, k);
                bool hit;
                normal64_fast(s.wk, r, hit);
                if (!hit && q >= s.p0 && q < s.p1)
                    out[(q - s.p0) / 64u] = normal64_continue(r, sub, s.K, q / 64u, s.wk);
            }
        }
    }
}

/* Device memory for the f64 normal fills of one context and device: the count of listed
 * misses, a copy of the layers and the list. */
struct NormalScratch {
    unsigned long long *count = nullptr;
    zig::Layer *wk = nullptr;
    NormalMiss *list = nullptr;
    uint64_t cap = 0;
    sycl::event last; /* the last work that uses it */
};

struct ScratchKey {
    sycl::context context;
    sycl::device device;
    bool operator==(const ScratchKey &) const = default;
};

struct ScratchHash {
    size_t operator()(const ScratchKey &k) const {
        return std::hash<sycl::context>()(k.context) * 31u + std::hash<sycl::device>()(k.device);
    }
};

inline std::mutex &normal_scratch_mutex() {
    static std::mutex m;
    return m;
}

/* The scratch of q's context and device with room for cap misses. The caller holds the mutex.
 * The memory lives to the end of the program: the SYCL runtime may be gone when static
 * destructors run. */
inline NormalScratch &normal_scratch(sycl::queue &q, uint64_t cap) {
    static auto *all = new std::unordered_map<ScratchKey, NormalScratch, ScratchHash>();
    NormalScratch &s = (*all)[ScratchKey{q.get_context(), q.get_device()}];
    if (s.count && s.cap >= cap)
        return s;
    if (s.count) {
        s.last.wait();
        sycl::free(s.count, q.get_context());
    }
    cap = std::max(cap, 2 * s.cap);
    char *m = static_cast<char *>(
        sycl::malloc_device(16 + sizeof zig::WK + cap * sizeof(NormalMiss), q));
    if (!m)
        throw std::bad_alloc();
    s.count = reinterpret_cast<unsigned long long *>(m);
    s.wk = reinterpret_cast<zig::Layer *>(m + 16);
    s.list = reinterpret_cast<NormalMiss *>(m + 16 + sizeof zig::WK);
    s.cap = cap;
    s.last = q.memcpy(s.wk, zig::WK, sizeof zig::WK);
    return s;
}

template <unsigned KD, bool SHIFTED, bool SHUFFLE, class Out>
sycl::event fill_normal64_table(sycl::queue &q, const Span &t, const Out &out, sycl::event zero,
                                NormalMiss *list, unsigned long long *count, uint64_t cap) {
    const uint64_t teams = (t.g1 - t.g0 + TILE_GROUPS) / TILE_GROUPS;
    return q.submit([&](sycl::handler &h) {
        h.depends_on(zero);
        auto dst = bind(out, h);
        sycl::local_accessor<NormalLocal<SHIFTED, SHUFFLE>, 1> lm(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>(teams * TILE_ITEMS, TILE_ITEMS), [=](sycl::nd_item<1> it) {
            normal64_octet_body<KD, SHIFTED, SHUFFLE>(
                dst.get(), t, it, *lm.template get_multi_ptr<sycl::access::decorated::no>().get(),
                list, count, cap);
        });
    });
}

template <bool SHUFFLE, class Out>
sycl::event fill_normal64_table(sycl::queue &q, const Span &t, const Out &out, sycl::event zero,
                                NormalMiss *list, unsigned long long *count, uint64_t cap) {
    const uint64_t D0 = t.p0 >> 6;
    if (D0 & 1u)
        return fill_normal64_table<1, true, SHUFFLE>(q, t, out, zero, list, count, cap);
    if (((D0 >> 1) & 7u) != 0)
        return fill_normal64_table<0, true, SHUFFLE>(q, t, out, zero, list, count, cap);
    return fill_normal64_table<0, false, SHUFFLE>(q, t, out, zero, list, count, cap);
}

/* The two kernels, with room for at most list_cap misses, n / 128 by default, which a fill
 * exceeds with negligible probability. The tests take a smaller list to run the walk. */
template <class Out>
sycl::event fill_normal64_list(sycl::queue &q, const Span &s, const Out &out,
                               const std::vector<sycl::event> &deps,
                               uint64_t list_cap = ~(uint64_t)0,
                               Exchange exchange = Exchange::Auto) {
    const uint64_t n = (s.p1 - s.p0) / 64u;
    const uint64_t want = std::min(n / 128u, list_cap);
    const uint64_t items = (n / 200u + TILE_ITEMS - 1u) / TILE_ITEMS * TILE_ITEMS;
    std::lock_guard<std::mutex> lock(normal_scratch_mutex());
    NormalScratch &sc = normal_scratch(q, want);
    std::vector<sycl::event> first = deps;
    first.push_back(sc.last);
    sycl::event zero = q.memset(sc.count, 0, sizeof *sc.count, first);
    Span t = s;
    t.wk = sc.wk;
    NormalMiss *list = sc.list;
    unsigned long long *count = sc.count;
    const bool shuffle = exchange == Exchange::Shuffle ||
                         (exchange == Exchange::Auto && subgroups_hold_groups(q.get_device()));
    sycl::event table = shuffle ? fill_normal64_table<true>(q, t, out, zero, list, count, want)
                                : fill_normal64_table<false>(q, t, out, zero, list, count, want);
    const Key sub = Rng::from_key(s.key, 0, s.K).sub(PURPOSE_NORMAL64).key();
    sc.last = q.submit([&](sycl::handler &h) {
        h.depends_on(table);
        auto dst = bind(out, h, true);
        h.parallel_for(sycl::range<1>(items), [=](sycl::item<1> it) {
            normal64_miss_body(dst.get(), t, sub, it.get_id(0), items, list, count, want);
        });
    });
    return sc.last;
}

/* Set the span of a fill that takes `n` elements of `bits` bits from the generator's position
 * aligned to `align` bits, and move the position past them. Returns false for an empty fill. */
inline bool plan_span(Rng &rng, uint64_t n, unsigned align, unsigned bits, Span &s) {
    uint64_t p0 = align_pos(rng.position(), align);
    if (p0 < rng.position() || n > (~(uint64_t)0 - p0) / bits)
        throw std::overflow_error("tandem::fill: the fill runs past stream position 2^64");
    s.K = rng.chunk_length();
    s.key = rng.key();
    s.p0 = p0;
    s.p1 = p0 + n * bits;
    s.range = s.thresh = 0;
    s.wk = nullptr; /* set by with_layers inside the kernel */
    s.table = ChoiceTable{};
    rng.set_position(s.p1);
    return n != 0;
}

/* The rows and groups that hold the blocks [ba, bb]. */
inline void set_rows(Span &s, uint64_t ba, uint64_t bb) {
    s.r0 = ba >> 3;
    s.r1 = bb >> 3;
    s.g0 = s.r0 / s.K;
    s.g1 = s.r1 / s.K;
}

template <class Kind, class Out>
sycl::event fill_kind(sycl::queue &q, const Out &out, uint64_t n, Rng &rng, Kernel kernel,
                      const std::vector<sycl::event> &deps, uint64_t range = 0,
                      const ChoiceTable &table = {}) {
    constexpr unsigned bits = elem<Kind>::bits;
    Span s;
    if (!plan_span(rng, n, bits, bits, s))
        return sycl::event();
    s.range = range;
    s.table = table;
    if constexpr (std::is_same_v<Kind, below32>)
        s.thresh = below_threshold_u32((uint32_t)range);
    else if constexpr (std::is_same_v<Kind, below64>)
        s.thresh = below_threshold_u64(range);
    set_rows(s, s.p0 >> 7, (s.p1 - 1) >> 7);
    if (kernel == Kernel::Auto)
        kernel = q.get_device().is_cpu() ? Kernel::Chunk : Kernel::Tile;
    if constexpr (std::is_same_v<Kind, normal64>) {
        if (kernel == Kernel::Tile && n >= NORMAL_LIST_MIN &&
            q.get_device().has(sycl::aspect::atomic64))
            return fill_normal64_list(q, s, out, deps);
    } else if (kernel == Kernel::Tile && s.K >= TILE_STEPS) {
        return fill_tile<Kind>(q, s, out, deps);
    }
    return fill_chunk<Kind>(q, s, out, deps);
}

/* The float Box-Muller step of the fills. The radius takes the logarithm of normal_block_f32 in
 * core.hpp, the angle the native cos and sin, as tandem-cuda takes __sincosf, shifted by half a
 * turn into [-pi, pi), where they are accurate, with both halves negated. Library logf and
 * sincos would make the fill compute bound on GPUs. The step stays in float on devices without
 * double precision, within 16 ulps + 1e-6 of box_muller2_f32. Define TANDEM_PRECISE_F32_NORMAL
 * for box_muller2_f32 itself. */
inline Pair2<float> normal_step_f32(float a, float b) {
#if defined(TANDEM_PRECISE_F32_NORMAL)
    return box_muller2_f32(a, b);
#else
    TANDEM_FP_NOCONTRACT
    float x = 1.0f - a;
    uint32_t ix = std::bit_cast<uint32_t>(x) + 0x004afb0du;
    float nk = (float)(127 - (int32_t)(ix >> 23));
    float mant = std::bit_cast<float>((ix & 0x007fffffu) + 0x3f3504f3u);
    float s = (mant - 1.0f) / (mant + 1.0f), zz = s * s;
    float p = sycl::fma(zz, 0.14275366f, 0.20000061f);
    p = sycl::fma(zz, p, 0.33333334f);
    p = sycl::fma(zz, p, 1.0f);
    float r = sycl::sqrt(
        sycl::fma(nk, 2.857213530660374e-06f, sycl::fma(nk, 1.38629150390625f, (s * -4.0f) * p)));
    float t = 6.2831853071795864769f * (b - 0.5f);
    return Pair2<float>{-r * sycl::native::cos(t), -r * sycl::native::sin(t)};
#endif
}

/* The float normal pair of the two Float32 draws in the words v. */
inline Pair2<float> normal_pair(const uint32_t *v) {
    return normal_step_f32(to_f32(v[0]), to_f32(v[1]));
}

/* Elements 2i and 2i + 1 of an n-element fill. `wide` says the pair sits at an 8-byte aligned
 * address. */
inline void store_pair(float *dst, bool wide, uint64_t n, uint64_t i, Pair2<float> z) {
    if (2 * i + 1 >= n) {
        dst[2 * i] = z.z0;
    } else if (wide) {
        const float two[2] = {z.z0, z.z1};
        store_wide<8>(dst + 2 * i, two);
    } else {
        dst[2 * i] = z.z0;
        dst[2 * i + 1] = z.z1;
    }
}

/* Float normal fill by Box-Muller pairs: pair j, the elements 2j and 2j + 1, comes from the
 * L = 2 32-bit stream slots that start at slot S + j L, S being the first slot of the fill, with
 * the cos half first. A work item owns the blocks that hold the last slot of a pair. s.range is
 * the element count, which an odd count leaves one past the last pair. A pair goes out as one
 * 8-byte store when the output allows it, and the two pairs of a block as one 16-byte store
 * when the fill starts on a block. With S % L != 0 an element can start in the previous block,
 * so the item also steps the chunk of that block, which for lane 0 is lane 7 one step back, or
 * at step 0 the previous group's last chunk. */
template <bool STRADDLE>
inline void normal_body(float *dst, const Span &s, uint64_t ba, uint64_t bb, uint64_t t) {
    constexpr unsigned L = 2;
    const uint64_t S = s.p0 >> 5, pairs = (s.p1 - s.p0) / (32u * L), n = s.range;
    const bool wide = (reinterpret_cast<uintptr_t>(dst) & 7u) == 0;
    const bool quad = !STRADDLE && (S & 3u) == 0 && (reinterpret_cast<uintptr_t>(dst) & 15u) == 0;
    uint64_t c = 8u * s.g0 + t, g = c >> 3, lane = c & 7u, row = g * s.K;
    uint32_t o[4], h[4], po[4] = {0, 0, 0, 0}, ph[4] = {0, 0, 0, 0};
    F_keyed(s.key.w, c, DOMAIN_STREAM, AUX_STREAM, o, h);
    if (STRADDLE)
        F_keyed(s.key.w, lane ? c - 1u : 8u * g + 7u, DOMAIN_STREAM, AUX_STREAM, po, ph);
    uint32_t j0 = row < s.r0 ? (uint32_t)(s.r0 - row) : 0u;
    uint32_t j1 = (uint32_t)(s.r1 - row < s.K - 1u ? s.r1 - row : s.K - 1u);
    for (uint32_t j = 0; j <= j1; j++) {
        T(o, h);
        if (STRADDLE && (lane || j))
            T(po, ph);
        uint64_t b = (row + j) * 8u + lane;
        if (j < j0 || b < ba || b > bb)
            continue;
        if (quad && 4u * (b - ba) + 4u <= n) {
            Pair2<float> z0 = normal_step_f32(to_f32(o[0]), to_f32(o[1]));
            Pair2<float> z1 = normal_step_f32(to_f32(o[2]), to_f32(o[3]));
            const float q4[4] = {z0.z0, z0.z1, z1.z0, z1.z1};
            store_wide<16>(dst + 4u * (b - ba), q4);
            continue;
        }
        if (!STRADDLE) {
            /* S % L == 0: the pairs of the block end at its halves and lie inside it. */
            for (unsigned m = 1; m < 4; m += L) {
                uint64_t last = 4u * b + m + 1u; /* one past the pair's last slot */
                if (last < S + L)
                    continue;
                uint64_t i = (last - S) / L - 1u;
                if (i < pairs)
                    store_pair(dst, wide, n, i, normal_pair(o + (m + 1u - L)));
            }
            continue;
        }
        /* Pairs can start in the previous block: x holds it, then this one. */
        uint32_t x[8] = {0, 0, 0, 0, o[0], o[1], o[2], o[3]};
        uint32_t q[4] = {0, 0, 0, 0};
        const uint32_t *prev = po;
        if (lane == 0 && j == 0) {
            if (g > 0)
                block(s.key.w, 8u * (g - 1u) + 7u, s.K - 1u, q);
            prev = q;
        }
        for (unsigned k = 0; k < 4; k++)
            x[k] = prev[k];
        for (unsigned m = 0; m < 4; m++) {
            uint64_t last = 4u * b + m + 1u;
            if (last < S + L || (last - S) % L)
                continue;
            uint64_t i = (last - S) / L - 1u;
            if (i < pairs)
                store_pair(dst, wide, n, i, normal_pair(x + (m + 5u - L)));
        }
    }
}

/* Unit u of a float normal fill from its four words: two pairs, 16 output bytes at element
 * 4u. */
inline void normal_unit(float *dst, bool wide, uint64_t n, uint64_t u, const uint32_t w[4]) {
    constexpr unsigned E = 4;
    float z[E];
    for (unsigned k = 0; k < E; k += 2) {
        Pair2<float> p = normal_pair(w + k);
        z[k] = p.z0;
        z[k + 1] = p.z1;
    }
    float *d = dst + E * u;
    if (wide && E * u + E <= n) {
        store_wide<16>(d, z);
    } else {
        for (unsigned k = 0; k < E; k++)
            if (E * u + k < n)
                d[k] = z[k];
    }
}

/* The local memory of normal_group_body: two steps of the blocks passed between lanes when
 * sub-groups cannot shuffle them, then the first row of each group and of the group after the
 * work group. A start at a block of lane 0 needs none, and less local memory lets more work
 * groups share a compute unit. */
constexpr unsigned NGROUP_XBUF = 2 * TILE_ITEMS, NGROUP_FIRST = TILE_ITEMS + 8;
template <bool SHIFTED, bool SHUFFLE>
constexpr unsigned ngroup_blocks = !SHIFTED  ? 1u
                                   : SHUFFLE ? NGROUP_FIRST
                                             : NGROUP_XBUF + NGROUP_FIRST;

/* Float normal fill for any start, each work item stepping only its own chunk. The fill is cut
 * into units of four 32-bit words from its first word S: unit u is two pairs, 16 output bytes at
 * element 4u, and starts at word KW = S % 4 of block B0 + u, B0 = S / 4. Lane l of a group writes unit 8m + l of octet m, so the eight lanes
 * write 128 aligned bytes whatever the start. With d = B0 % 8 the octet's units start in row R
 * from lane d on and in row R + 1 below lane d, and each ends in the next block. So a group
 * writes the octet of row R one step later, when row R + 1 is out. Each step every lane takes
 * word t of the new row from lane la (t >= KW, its unit's first block) or lb (t < KW, its
 * second), by sub-group shuffles with SHUFFLE, else through local memory with a barrier per
 * step, and keeps what it took the step before, which is row R. The octet of a group's last row
 * takes the next group's first row, kept from step 0, or for the group after the work group
 * computed in one step. KW and SHIFTED (KW != 0 or d != 0) are template parameters because
 * shifts by a runtime KW, or both paths in one kernel, cost a sixth of the float fill's speed. */
template <unsigned KW, bool SHIFTED, bool SHUFFLE>
inline void normal_group_body(float *dst, const Span &s, sycl::nd_item<1> it, Block *lm) {
    constexpr unsigned E = 4;
    Block *first = lm + (SHUFFLE ? 0u : NGROUP_XBUF);
    const uint64_t S = s.p0 >> 5, n = s.range, units = (n + E - 1u) / E, end = S + 4u * units;
    const uint64_t B0 = S >> 2, R0 = B0 >> 3; /* octet m lies at rows R0 + m and R0 + m + 1 */
    const unsigned d = (unsigned)(B0 & 7u);
    const bool wide = (reinterpret_cast<uintptr_t>(dst) & 15u) == 0;
    const uint64_t gb = s.g0 + (uint64_t)it.get_group(0) * TILE_GROUPS;
    const unsigned rank = (unsigned)it.get_local_id(0), gi = rank >> 3, lane = rank & 7u;
    const uint64_t g = gb + gi;
    const bool mine = g <= s.g1;
    /* The lanes of this lane's unit's first and second block, and whether each lies in R + 1. */
    const unsigned la = (d + lane) & 7u, lb = (d + lane + 1u) & 7u;
    const bool na = d + lane >= 8u, nb = d + lane + 1u >= 8u;
    uint32_t o[4], h[4];
    F_keyed(s.key.w, 8u * g + lane, DOMAIN_STREAM, AUX_STREAM, o, h);
    if (SHIFTED && rank < 8u)
        block(s.key.w, 8u * (gb + TILE_GROUPS) + rank, 0, first[TILE_ITEMS + rank].w);

    /* Unit 8 (R - R0) + lane: word t is word (t + KW) % 4 of its first block (t + KW < 4) or of
     * its second, taken from row R + 1 (late) or row R (early). */
    auto emit = [&](uint64_t R, const uint32_t *late, const uint32_t *early) {
        if (R < R0 || 8u * (R - R0) + lane >= units)
            return;
        uint32_t w[4];
        for (unsigned t = 0; t < 4; t++) {
            const unsigned k = (t + KW) & 3u;
            w[t] = (t + KW < 4u ? na : nb) ? late[k] : early[k];
        }
        normal_unit(dst, wide, n, 8u * (R - R0) + lane, w);
    };
    /* Word t of the block of lane la (t >= KW) or lb (t < KW). Every work item calls it. */
    auto fetch = [&](unsigned buf, uint32_t *x) {
        if constexpr (SHUFFLE) {
            (void)buf;
            auto sg = it.get_sub_group();
            unsigned base = (unsigned)sg.get_local_linear_id() - lane;
            for (unsigned t = 0; t < 4; t++)
                x[t] = sycl::select_from_group(sg, o[t], base + (t >= KW ? la : lb));
        } else {
            Block *xa = lm + buf * TILE_ITEMS;
            xa[rank] = Block{{o[0], o[1], o[2], o[3]}};
            sycl::group_barrier(it.get_group());
            for (unsigned t = 0; t < 4; t++)
                x[t] = xa[rank - lane + (t >= KW ? la : lb)].w[t];
        }
    };

    /* Whether step j has a row in the fill, the same for the whole work group. */
    auto live = [&](uint32_t j) { return j < s.K && (gb * s.K + j) * 32u < end; };
    if constexpr (!SHIFTED) {
        /* Units are the blocks themselves. */
        for (uint32_t j = 0; live(j); j++)
            if (mine) {
                T(o, h);
                emit(g * s.K + j, o, o);
            }
    } else {
        /* Step 0 is peeled so that the loop holds no test for it. */
        uint32_t j = 0, prev[4] = {0, 0, 0, 0};
        if (live(0)) {
            if (mine)
                T(o, h);
            first[gi * 8u + lane] = Block{{o[0], o[1], o[2], o[3]}};
            fetch(0, prev);
            for (j = 1; live(j); j++) {
                if (mine)
                    T(o, h);
                uint32_t cur[4];
                fetch(j & 1u, cur);
                if (mine)
                    emit(g * s.K + j - 1u, cur, prev);
                for (unsigned t = 0; t < 4; t++)
                    prev[t] = cur[t];
            }
        }
        /* The octet of the last row stepped. Its row R + 1 is the next group's first row after
         * a whole group, and after a loop cut short lies past the fill, so the units that need
         * it are past the fill too. */
        sycl::group_barrier(it.get_group());
        const Block *next = first + (gi + 1u) * 8u;
        uint32_t cur[4];
        for (unsigned t = 0; t < 4; t++)
            cur[t] = next[t >= KW ? la : lb].w[t];
        if (mine && j > 0)
            emit(g * s.K + j - 1u, cur, prev);
    }
}

template <unsigned KW, bool SHIFTED, bool SHUFFLE, class Out>
sycl::event fill_normal_group_kw(sycl::queue &q, const Span &s, const Out &out,
                                 const std::vector<sycl::event> &deps) {
    uint64_t teams = (s.g1 - s.g0 + TILE_GROUPS) / TILE_GROUPS;
    return q.submit([&](sycl::handler &h) {
        h.depends_on(deps);
        auto dst = bind(out, h);
        sycl::local_accessor<Block, 1> lm(sycl::range<1>(ngroup_blocks<SHIFTED, SHUFFLE>), h);
        h.parallel_for(sycl::nd_range<1>(teams * TILE_ITEMS, TILE_ITEMS), [=](sycl::nd_item<1> it) {
            normal_group_body<KW, SHIFTED, SHUFFLE>(
                dst.get(), s, it, lm.template get_multi_ptr<sycl::access::decorated::no>().get());
        });
    });
}

template <bool SHUFFLE, class Out>
sycl::event fill_normal_group(sycl::queue &q, const Span &s, const Out &out,
                              const std::vector<sycl::event> &deps) {
    switch ((s.p0 >> 5) & 3u) {
    case 0:
        if (((s.p0 >> 7) & 7u) == 0)
            return fill_normal_group_kw<0, false, SHUFFLE>(q, s, out, deps);
        return fill_normal_group_kw<0, true, SHUFFLE>(q, s, out, deps);
    case 1:
        return fill_normal_group_kw<1, true, SHUFFLE>(q, s, out, deps);
    case 2:
        return fill_normal_group_kw<2, true, SHUFFLE>(q, s, out, deps);
    default:
        return fill_normal_group_kw<3, true, SHUFFLE>(q, s, out, deps);
    }
}

template <class Out>
sycl::event fill_normal_f32_kind(sycl::queue &q, const Out &out, uint64_t n, Rng &rng,
                                 const std::vector<sycl::event> &deps,
                                 Kernel kernel = Kernel::Auto,
                                 Exchange exchange = Exchange::Auto) {
    constexpr unsigned L = 2;
    if (n == 0) /* no draws, so no alignment either */
        return sycl::event();
    uint64_t pairs = (n + 1u) / 2u;
    Span s;
    plan_span(rng, pairs, 32u, 32u * L, s);
    s.range = n;
    uint64_t S = s.p0 >> 5, ba = (S + L - 1u) >> 2, bb = (S + pairs * L - 1u) >> 2;
    set_rows(s, ba, bb);
    if (kernel == Kernel::Auto)
        kernel = q.get_device().is_cpu() ? Kernel::Chunk : Kernel::Tile;
    if (kernel == Kernel::Tile) {
        /* The units cover the blocks from the fill's first word to the last unit's last. */
        set_rows(s, S >> 2, (S + 4u * ((n + 3u) / 4u) - 1u) >> 2);
        if (exchange == Exchange::Shuffle ||
            (exchange == Exchange::Auto && subgroups_hold_groups(q.get_device())))
            return fill_normal_group<true>(q, s, out, deps);
        return fill_normal_group<false>(q, s, out, deps);
    }
    bool straddle = S % L != 0;
    return q.submit([&](sycl::handler &h) {
        h.depends_on(deps);
        auto dst = bind(out, h);
        h.parallel_for(sycl::range<1>(8u * (s.g1 - s.g0 + 1u)), [=](sycl::item<1> it) {
            if (straddle)
                normal_body<true>(dst.get(), s, ba, bb, it.get_id(0));
            else
                normal_body<false>(dst.get(), s, ba, bb, it.get_id(0));
        });
    });
}

/* Float normals by Box-Muller, double normals by the ziggurat, one u64 draw per element. */
template <class E, class Out>
sycl::event fill_normal_kind(sycl::queue &q, const Out &out, uint64_t n, Rng &rng,
                             const std::vector<sycl::event> &deps, Kernel kernel = Kernel::Auto) {
    static_assert(std::is_same_v<E, float> || std::is_same_v<E, double>,
                  "tandem::fill_normal: the value type must be float or double");
    if constexpr (std::is_same_v<E, double>)
        return fill_kind<normal64>(q, out, n, rng, kernel, deps);
    else
        return fill_normal_f32_kind(q, out, n, rng, deps, kernel);
}

/* What a value type draws: the kind of fill, and how many of its elements one value takes.
 * Signed integers read the unsigned draw of the same width in two's complement, a complex
 * value the real then the imaginary component. */
template <unsigned N> struct uint_of;
template <> struct uint_of<1> {
    using type = uint8_t;
};
template <> struct uint_of<2> {
    using type = uint16_t;
};
template <> struct uint_of<4> {
    using type = uint32_t;
};
template <> struct uint_of<8> {
    using type = uint64_t;
};

template <class E, class = void> struct plan;
template <> struct plan<bool> {
    using kind = bool;
    static constexpr unsigned parts = 1;
};
template <class E>
struct plan<E, std::enable_if_t<std::is_integral_v<E> && !std::is_same_v<E, bool>>> {
    using kind = typename uint_of<sizeof(E)>::type;
    static constexpr unsigned parts = 1;
};
template <> struct plan<float> {
    using kind = float;
    static constexpr unsigned parts = 1;
};
template <> struct plan<double> {
    using kind = double;
    static constexpr unsigned parts = 1;
};
template <> struct plan<sycl::half> {
    static_assert(sizeof(sycl::half) == 2, "unexpected sycl::half layout");
    using kind = f16_bits;
    static constexpr unsigned parts = 1;
};
template <class T> struct plan<std::complex<T>, std::enable_if_t<std::is_floating_point_v<T>>> {
    static_assert(sizeof(std::complex<T>) == 2 * sizeof(T), "unexpected complex layout");
    using kind = T;
    static constexpr unsigned parts = 2;
};

template <class E> using out_of = typename elem<typename plan<E>::kind>::out_t;

} // namespace detail

/* Fill n elements of USM memory that the queue's device can write with the draws that start
 * at the generator's position, the values the specification's fill defines, and move the
 * position past them. The value type is bool, an integer of 8 to 64 bits, float, double,
 * sycl::half, std::complex<float> or std::complex<double>. Signed integers hold the two's
 * complement of the unsigned draw. A complex value takes two draws, the real and then the
 * imaginary component. A uint16_t output gets raw draws: use fill_f16_bits for Float16 bit
 * patterns.
 *
 * The kernel waits for `deps` and is not waited for, but the generator advances at once, so
 * the next fill or draw continues the stream. An empty fill aligns the position to the width
 * and returns a complete event. */
template <class E>
sycl::event fill(sycl::queue &q, E *out, size_t n, Rng &rng,
                 const std::vector<sycl::event> &deps = {}) {
    using P = detail::plan<E>;
    return detail::fill_kind<typename P::kind>(
        q, detail::UsmOut<detail::out_of<E>>{reinterpret_cast<detail::out_of<E> *>(out)},
        (uint64_t)n * P::parts, rng, detail::Kernel::Auto, deps);
}

/* The same into a whole buffer of any rank, in its linear order. */
template <class E, int D> sycl::event fill(sycl::queue &q, sycl::buffer<E, D> &buf, Rng &rng) {
    using P = detail::plan<E>;
    return detail::fill_kind<typename P::kind>(q, detail::BufOut<detail::out_of<E>, E, D>{buf},
                                               (uint64_t)buf.size() * P::parts, rng,
                                               detail::Kernel::Auto, {});
}

/* Binary16 bit patterns of the specification's Float16 draws, (raw >> 5) * 2^-11. */
inline sycl::event fill_f16_bits(sycl::queue &q, uint16_t *out, size_t n, Rng &rng,
                                 const std::vector<sycl::event> &deps = {}) {
    return detail::fill_kind<detail::f16_bits>(q, detail::UsmOut<uint16_t>{out}, n, rng,
                                               detail::Kernel::Auto, deps);
}
template <int D>
sycl::event fill_f16_bits(sycl::queue &q, sycl::buffer<uint16_t, D> &buf, Rng &rng) {
    return detail::fill_kind<detail::f16_bits>(q, detail::BufOut<uint16_t, uint16_t, D>{buf},
                                               buf.size(), rng, detail::Kernel::Auto, {});
}

namespace detail {
template <class E> struct below {
    static_assert(std::is_unsigned_v<E> && !std::is_same_v<E, bool> &&
                      (sizeof(E) == 4 || sizeof(E) == 8),
                  "tandem::fill_below: the value type must be an unsigned 32- or 64-bit integer");
    using kind = std::conditional_t<sizeof(E) == 4, below32, below64>;
    using out_t = typename elem<kind>::out_t;
};
} // namespace detail

/* Uniform integers on [0, range) in an unsigned 32- or 64-bit integer type, by Lemire's method
 * as Rng::urand(range). Element i takes draw i of the u32 (u64) fill and consumes exactly that
 * one draw, so the fill advances the position by 32 n (64 n) bits whatever the draws are. A
 * rejected draw retries on the fallback stream split(g) of sub(PURPOSE_BELOW32) (or 64) of the
 * key at position 0, g being the draw's global index, aligned start / 32 (64) + i. An empty fill
 * leaves the position alone. Appendix A of the specification. */
template <class E>
sycl::event fill_below(sycl::queue &q, E *out, size_t n, Rng &rng, std::type_identity_t<E> range,
                       const std::vector<sycl::event> &deps = {}) {
    using B = detail::below<E>;
    if (n == 0) /* no draws, so no alignment either */
        return sycl::event();
    return detail::fill_kind<typename B::kind>(
        q, detail::UsmOut<typename B::out_t>{reinterpret_cast<typename B::out_t *>(out)}, n, rng,
        detail::Kernel::Auto, deps, range);
}
template <class E, int D>
sycl::event fill_below(sycl::queue &q, sycl::buffer<E, D> &buf, Rng &rng,
                       std::type_identity_t<E> range) {
    using B = detail::below<E>;
    if (buf.size() == 0)
        return sycl::event();
    return detail::fill_kind<typename B::kind>(q, detail::BufOut<typename B::out_t, E, D>{buf},
                                               buf.size(), rng, detail::Kernel::Auto, {}, range);
}

/* Standard normals in float or double, Appendix A of the specification.
 *
 * Double: the 1024-layer ziggurat, the sequence of Rng::normal calls. Element i takes draw i of
 * the u64 fill and consumes exactly that draw. A draw that misses the fast path (0.43 %)
 * continues on split(g) of sub(PURPOSE_NORMAL64) of the key at position 0, g being the draw's
 * global index, aligned start / 64 + i, so a fill cut at any element equals the whole fill. An
 * empty fill aligns the position to 64 bits. The values equal tandem-c's bit for bit.
 *
 * Float: Box-Muller, the flattened sequence of Rng::normalf2 calls. Pair j, the elements 2j and
 * 2j + 1 with the cos half first, is made from the draws 2j and 2j + 1 of the f32 fill. An odd
 * count drops the last sin half and still consumes both draws. An empty fill leaves the position
 * alone. The step takes native cos and sin and agrees with other ports to 16 ulps + 1e-6. */
template <class E>
sycl::event fill_normal(sycl::queue &q, E *out, size_t n, Rng &rng,
                        const std::vector<sycl::event> &deps = {}) {
    return detail::fill_normal_kind<E>(q, detail::UsmOut<E>{out}, n, rng, deps);
}
template <class E, int D>
sycl::event fill_normal(sycl::queue &q, sycl::buffer<E, D> &buf, Rng &rng) {
    return detail::fill_normal_kind<E>(q, detail::BufOut<E, E, D>{buf}, buf.size(), rng, {});
}

namespace detail {
template <class E> struct exponential {
    static_assert(std::is_same_v<E, float> || std::is_same_v<E, double>,
                  "tandem::fill_exponential: the value type must be float or double");
    using kind = std::conditional_t<std::is_same_v<E, float>, exp32, exp64>;
};
} // namespace detail

/* Standard exponentials -log(1 - u) in float or double, the sequence of Rng::exponentialf or
 * Rng::exponential calls: element i is made from draw i of the f32 or f64 fill. The logarithm is
 * the host normals' polynomial with explicit fused multiply-adds and IEEE division, so the values
 * equal tandem-c's bit for bit. An empty fill leaves the position alone. Appendix A of the
 * specification. */
template <class E>
sycl::event fill_exponential(sycl::queue &q, E *out, size_t n, Rng &rng,
                             const std::vector<sycl::event> &deps = {}) {
    if (n == 0) /* no draws, so no alignment either */
        return sycl::event();
    return detail::fill_kind<typename detail::exponential<E>::kind>(
        q, detail::UsmOut<E>{out}, n, rng, detail::Kernel::Auto, deps);
}
template <class E, int D>
sycl::event fill_exponential(sycl::queue &q, sycl::buffer<E, D> &buf, Rng &rng) {
    if (buf.size() == 0)
        return sycl::event();
    return detail::fill_kind<typename detail::exponential<E>::kind>(
        q, detail::BufOut<E, E, D>{buf}, buf.size(), rng, detail::Kernel::Auto, {});
}

/* Weighted choice indices, Appendix C of the specification: element i maps UInt64 draw i of the
 * fill through the alias table, so the fill consumes 64 bits per element, never retries, equals
 * the Rng::choice calls and is bit identical to tandem-c's tandem_fill_choice. Build the table
 * with choice_build on the host, then copy its cut and alias arrays to USM memory that the
 * queue's device reads, and view them with a ChoiceTable. An empty fill aligns the position to
 * 64 bits. */
inline sycl::event fill_choice(sycl::queue &q, uint32_t *out, size_t n, Rng &rng,
                               const ChoiceTable &table,
                               const std::vector<sycl::event> &deps = {}) {
    return detail::fill_kind<detail::choice_idx>(q, detail::UsmOut<uint32_t>{out}, n, rng,
                                                 detail::Kernel::Auto, deps, 0, table);
}
template <int D>
sycl::event fill_choice(sycl::queue &q, sycl::buffer<uint32_t, D> &buf, Rng &rng,
                        const ChoiceTable &table) {
    return detail::fill_kind<detail::choice_idx>(q, detail::BufOut<uint32_t, uint32_t, D>{buf},
                                                 buf.size(), rng, detail::Kernel::Auto, {}, 0,
                                                 table);
}

} // namespace tandem
