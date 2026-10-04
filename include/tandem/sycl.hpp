/* Tandem8x32 for SYCL: fills of USM memory and buffers on any SYCL device, and the scalar
 * generator tandem::Rng from tandem/core.hpp for draws inside kernels. Header only, C++20.
 *
 * Implements https://github.com/tandem-rng/spec and produces the stream it defines, bit for
 * bit. Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
 */
#pragma once

#include <sycl/sycl.hpp>

#include <bit>
#include <complex>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
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

/* The fill's geometry: stream bits [p0, p1), rows r0 .. r1 inclusive, groups g0 .. g1, and
 * what the bounded kinds need: the fill's key and chunk length, the range and its rejection
 * threshold. The normal fill keeps its element count in `range`. */
struct Span {
    uint64_t p0, p1, r0, r1, g0, g1;
    uint32_t K;
    Key key;
    uint64_t range, thresh;
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

/* Whether every block of the fill lands on a 16-byte address: the output's first byte and the
 * fill's first stream byte (bit, for bool) agree modulo 16. */
template <class Kind> bool blocks_aligned(const typename elem<Kind>::out_t *out, const Span &s) {
    uint64_t first = std::is_same_v<Kind, bool> ? s.p0 : s.p0 / 8;
    return ((reinterpret_cast<uintptr_t>(out) - first) & 15u) == 0;
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

/* The elements of the block at stream bit P, by Lemire's method for the bounded kinds, as
 * below_u32_t and below_u64_t in core.hpp. Those accept or flag every draw of the block first
 * and then retry the flagged ones in one loop, because a device compiler that inlines
 * everything would otherwise copy the retry, a generator's seeding, once per element. The
 * retry of draw k takes the global draw index (P + k bits) / bits, as Appendix A of the
 * specification says, so a fill cut into pieces equals the whole fill. */
template <class Kind>
inline void block_values(const uint32_t w[4], uint64_t P, const Span &s,
                         typename elem<Kind>::out_t x[]) {
    using O = typename elem<Kind>::out_t;
    constexpr unsigned bits = elem<Kind>::bits, per_block = 128 / bits;
    if constexpr (is_below<Kind>) {
        const O range = (O)s.range, t = (O)s.thresh;
        unsigned rejected = 0;
        for (unsigned k = 0; k < per_block; k++) {
            O u = elem<Kind>::draw(w, k), lo;
            if constexpr (sizeof(O) == 4) {
                uint64_t m = (uint64_t)u * range;
                x[k] = (O)(m >> 32);
                lo = (O)m;
            } else {
                x[k] = mulhi64(u, range);
                lo = u * range;
            }
            if (lo < t)
                rejected |= 1u << k;
        }
        while (rejected) {
            unsigned k = (unsigned)std::countr_zero(rejected);
            rejected &= rejected - 1u;
            uint64_t g = P / bits + k;
            O v;
            if constexpr (sizeof(O) == 4)
                v = below_retry_u32(range, t, s.key.w, s.K, g);
            else
                v = below_retry_u64(range, t, s.key.w, s.K, g);
            /* A select per element, since a variable index would put x in local memory. */
            for (unsigned j = 0; j < per_block; j++)
                if (j == k)
                    x[j] = v;
        }
    } else {
        for (unsigned k = 0; k < per_block; k++)
            x[k] = elem<Kind>::make(w, k);
    }
}

/* Store the elements of the block at stream bit P that fall inside the fill's bits [p0, p1).
 * With ALIGNED a block fully inside leaves as 16-byte stores: one, or eight for bool, whose
 * 128 elements are one byte each. */
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
            store_wide<16>(out + (P - s.p0) / bits, x);
            return;
        }
        for (unsigned k = 0; k < per_block; k++) {
            uint64_t q = P + k * bits;
            if (q >= s.p0 && q < s.p1)
                out[(q - s.p0) / bits] = x[k];
        }
    }
}

/* Auto is Tile on GPUs and other accelerators with K >= 8, else Chunk. */
enum class Kernel {
    Auto,
    Chunk, /* one work item per chunk, one block per step */
    Tile,  /* one work group per 32 groups, eight steps staged in local memory */
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

template <class O> UsmOut<O> bind(const UsmOut<O> &u, sycl::handler &) { return u; }
template <class O, class T, int D>
AccOut<O, T, D> bind(const BufOut<O, T, D> &b, sycl::handler &h) {
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

template <class Kind, class Out>
sycl::event fill_chunk(sycl::queue &q, const Span s, const Out &out,
                       const std::vector<sycl::event> &deps) {
    return q.submit([&](sycl::handler &h) {
        h.depends_on(deps);
        auto dst = bind(out, h);
        h.parallel_for(sycl::range<1>(8u * (s.g1 - s.g0 + 1u)), [=](sycl::item<1> it) {
            auto *p = dst.get();
            if (blocks_aligned<Kind>(p, s))
                chunk_body<Kind, true>(p, s, it.get_id(0));
            else
                chunk_body<Kind, false>(p, s, it.get_id(0));
        });
    });
}

constexpr unsigned TILE_ITEMS = 256, TILE_GROUPS = TILE_ITEMS / 8, TILE_STEPS = 8;
constexpr unsigned TILE_SLOTS = TILE_GROUPS * TILE_STEPS * 8;

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
                      const std::vector<sycl::event> &deps, uint64_t range = 0) {
    constexpr unsigned bits = elem<Kind>::bits;
    Span s;
    if (!plan_span(rng, n, bits, bits, s))
        return sycl::event();
    s.range = range;
    if constexpr (std::is_same_v<Kind, below32>)
        s.thresh = below_threshold_u32((uint32_t)range);
    else if constexpr (std::is_same_v<Kind, below64>)
        s.thresh = below_threshold_u64(range);
    set_rows(s, s.p0 >> 7, (s.p1 - 1) >> 7);
    if (kernel == Kernel::Auto)
        kernel = q.get_device().is_cpu() ? Kernel::Chunk : Kernel::Tile;
    if (kernel == Kernel::Tile && s.K >= TILE_STEPS)
        return fill_tile<Kind>(q, s, out, deps);
    return fill_chunk<Kind>(q, s, out, deps);
}

/* The float Box-Muller step of the fill. sycl::sincos takes the angle shifted by half a turn,
 * which lies in [-pi, pi) and is rounded half as coarsely as 2 pi b, and both halves change
 * sign. This keeps the fill in float on devices without double precision, within 16 ulps + 1e-6
 * of box_muller2_f32. Define TANDEM_PRECISE_F32_NORMAL for box_muller2_f32 itself. */
inline Pair2<float> normal_step_f32(float a, float b) {
#if defined(TANDEM_PRECISE_F32_NORMAL)
    return box_muller2_f32(a, b);
#else
    float r = sycl::sqrt(-2.0f * sycl::log(1.0f - a)), c;
    float s = sycl::sincos(6.2831853071795864769f * (b - 0.5f), &c);
    return Pair2<float>{-r * c, -r * s};
#endif
}

/* The double step. core.hpp's box_muller2 is the host polynomial outside CUDA and HIP, an out
 * of line call per pair, so a kernel takes the device's log and sincos instead. */
inline Pair2<double> normal_step_f64(double a, double b) {
    double r = sycl::sqrt(-2.0 * sycl::log(1.0 - a)), c;
    double s = sycl::sincos(6.283185307179586 * b, &c);
    return Pair2<double>{r * c, r * s};
}

/* The normal pair of the uniforms in the words v: two Float64 draws (four words) or two Float32
 * draws (two words). */
template <class O> inline Pair2<O> normal_pair(const uint32_t *v) {
    if constexpr (std::is_same_v<O, double>)
        return normal_step_f64(to_f64(v[0] | ((uint64_t)v[1] << 32)),
                               to_f64(v[2] | ((uint64_t)v[3] << 32)));
    else
        return normal_step_f32(to_f32(v[0]), to_f32(v[1]));
}

/* Elements 2i and 2i + 1 of an n-element fill. `wide` says the pair sits at a 2 sizeof(O)
 * aligned address. */
template <class O> inline void store_pair(O *dst, bool wide, uint64_t n, uint64_t i, Pair2<O> z) {
    if (2 * i + 1 >= n) {
        dst[2 * i] = z.z0;
    } else if (wide) {
        const O two[2] = {z.z0, z.z1};
        store_wide<2 * sizeof(O)>(dst + 2 * i, two);
    } else {
        dst[2 * i] = z.z0;
        dst[2 * i + 1] = z.z1;
    }
}

/* Normal fill by Box-Muller pairs: pair j, the elements 2j and 2j + 1, comes from the L = 2
 * (float) or 4 (double) 32-bit stream slots that start at slot S + j L, S being the first slot
 * of the fill, with the cos half first. A work item owns the blocks that hold the last slot of
 * a pair. s.range is the element count, which an odd count leaves one past the last pair. A
 * pair goes out as one 8-byte (float) or 16-byte (double) store when the output allows it, and
 * the two float pairs of a block as one 16-byte store when the fill starts on a block. With
 * S % L != 0 an element can start in the previous block, so the item also steps the chunk of
 * that block, which for lane 0 is lane 7 one step back, or at step 0 the previous group's last
 * chunk. */
template <class O, bool STRADDLE>
inline void normal_body(O *dst, const Span &s, uint64_t ba, uint64_t bb, uint64_t t) {
    constexpr unsigned L = std::is_same_v<O, double> ? 4u : 2u;
    const uint64_t S = s.p0 >> 5, pairs = (s.p1 - s.p0) / (32u * L), n = s.range;
    const bool wide = (reinterpret_cast<uintptr_t>(dst) & (2 * sizeof(O) - 1)) == 0;
    const bool quad =
        !STRADDLE && L == 2 && (S & 3u) == 0 && (reinterpret_cast<uintptr_t>(dst) & 15u) == 0;
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
            /* S % L == 0: the pairs of the block end at its halves (float) or its end
             * (double), and lie inside it. */
            for (unsigned m = L == 2 ? 1u : 3u; m < 4; m += L) {
                uint64_t last = 4u * b + m + 1u; /* one past the pair's last slot */
                if (last < S + L)
                    continue;
                uint64_t i = (last - S) / L - 1u;
                if (i < pairs)
                    store_pair<O>(dst, wide, n, i, normal_pair<O>(o + (m + 1u - L)));
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
                store_pair<O>(dst, wide, n, i, normal_pair<O>(x + (m + 5u - L)));
        }
    }
}

template <class O, class Out>
sycl::event fill_normal_kind(sycl::queue &q, const Out &out, uint64_t n, Rng &rng,
                             const std::vector<sycl::event> &deps) {
    constexpr unsigned L = std::is_same_v<O, double> ? 4u : 2u;
    if (n == 0) /* no draws, so no alignment either */
        return sycl::event();
    uint64_t pairs = (n + 1u) / 2u;
    Span s;
    plan_span(rng, pairs, 32u * L / 2u, 32u * L, s);
    s.range = n;
    uint64_t S = s.p0 >> 5, ba = (S + L - 1u) >> 2, bb = (S + pairs * L - 1u) >> 2;
    set_rows(s, ba, bb);
    bool straddle = S % L != 0;
    return q.submit([&](sycl::handler &h) {
        h.depends_on(deps);
        auto dst = bind(out, h);
        h.parallel_for(sycl::range<1>(8u * (s.g1 - s.g0 + 1u)), [=](sycl::item<1> it) {
            if (straddle)
                normal_body<O, true>(dst.get(), s, ba, bb, it.get_id(0));
            else
                normal_body<O, false>(dst.get(), s, ba, bb, it.get_id(0));
        });
    });
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

/* Standard normals in float or double by Box-Muller, the flattened sequence of Rng::normalf2
 * or Rng::normal2 calls: pair j, the elements 2j and 2j + 1 with the cos half first, is made from
 * the draws 2j and 2j + 1 of the f32 or f64 fill. An odd count drops the last sin half and still
 * consumes both draws, so the fill takes 64 (float) or 128 (double) bits per pair. An empty fill
 * leaves the position alone. Device and host trigonometry differ in the last bits, so normals
 * agree with other ports to 16 ulps + 1e-6 (float) and 1e-12 relative (double), not bit for
 * bit. Not part of the specification. */
template <class E>
sycl::event fill_normal(sycl::queue &q, E *out, size_t n, Rng &rng,
                        const std::vector<sycl::event> &deps = {}) {
    static_assert(std::is_same_v<E, float> || std::is_same_v<E, double>,
                  "tandem::fill_normal: the value type must be float or double");
    return detail::fill_normal_kind<E>(q, detail::UsmOut<E>{out}, n, rng, deps);
}
template <class E, int D>
sycl::event fill_normal(sycl::queue &q, sycl::buffer<E, D> &buf, Rng &rng) {
    static_assert(std::is_same_v<E, float> || std::is_same_v<E, double>,
                  "tandem::fill_normal: the value type must be float or double");
    return detail::fill_normal_kind<E>(q, detail::BufOut<E, E, D>{buf}, buf.size(), rng, {});
}

} // namespace tandem
