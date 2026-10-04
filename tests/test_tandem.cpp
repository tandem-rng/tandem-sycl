// Spec vectors, reference stream dumps, fills against in-kernel draws and the u32 stream, split
// fills, derived keys, bounded and normal fills against the tandem-cuda fixtures, buffers, and
// position advancement, on the default SYCL device.
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>

#include <random>
#include <string>
#include <vector>

#include <tandem/sycl.hpp>

#include "../external/tandem-cuda/tests/cross_fill_below.h"
#include "../external/tandem-cuda/tests/cross_fill_normal.h"
#include "vectors.hpp"

using tandem::Key;
using tandem::Rng;
using tandem::detail::Kernel;

// A kernel captures the generator by value, so its size is part of the contract.
static_assert(sizeof(Rng) == 80, "Rng must stay 80 bytes");

static long checks, failures;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        checks++;                                                                                  \
        if (!(cond)) {                                                                             \
            failures++;                                                                            \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                            \
        }                                                                                          \
    } while (0)

static Key key_of(const uint32_t w[4]) { return Key{{w[0], w[1], w[2], w[3]}}; }
static bool words_equal(const uint32_t a[4], const uint32_t b[4]) {
    return std::memcmp(a, b, 16) == 0;
}

template <class E> using kind_of = typename tandem::detail::plan<E>::kind;
template <class E> constexpr unsigned bits_of = tandem::detail::elem<kind_of<E>>::bits;

// Host copies and dumps hold bools as bytes, because std::vector<bool> has no data().
template <class E> using host_t = std::conditional_t<std::is_same_v<E, bool>, uint8_t, E>;

static const Kernel KERNELS[] = {Kernel::Chunk, Kernel::Tile};
static const char *name(Kernel k) { return k == Kernel::Tile ? "tile" : "chunk"; }

// Device memory of n elements, freed on scope exit.
template <class T> struct Dev {
    sycl::queue &q;
    T *p;
    Dev(sycl::queue &q, size_t n) : q(q), p(sycl::malloc_device<T>(n ? n : 1, q)) {}
    ~Dev() { sycl::free(p, q); }
    Dev(const Dev &) = delete;
    std::vector<host_t<T>> host(size_t n, size_t from = 0) const {
        std::vector<host_t<T>> v(n);
        if (n)
            q.memcpy(v.data(), p + from, n * sizeof(T)).wait();
        return v;
    }
};

// Host copy of a fill of n elements from (key, pos, K) with an explicit kernel, `shift`
// elements into the allocation so that the output's alignment varies.
template <class E>
static std::vector<host_t<E>> device_fill(sycl::queue &q, const Key &key, uint64_t pos,
                                          uint32_t K, size_t n, Kernel kernel, size_t shift = 0,
                                          uint64_t *end = nullptr) {
    using O = tandem::detail::out_of<E>;
    Dev<E> d(q, n + 4);
    Rng r = Rng::from_key(key, pos, K);
    tandem::detail::fill_kind<kind_of<E>>(
        q, tandem::detail::UsmOut<O>{reinterpret_cast<O *>(d.p + shift)},
        (uint64_t)n * tandem::detail::plan<E>::parts, r, kernel, {})
        .wait();
    if (end)
        *end = r.position();
    return d.host(n, shift);
}

static std::vector<uint16_t> device_f16_bits(sycl::queue &q, const Key &key, uint64_t pos,
                                             uint32_t K, size_t n, Kernel kernel) {
    Dev<uint16_t> d(q, n);
    Rng r = Rng::from_key(key, pos, K);
    tandem::detail::fill_kind<tandem::detail::f16_bits>(q, tandem::detail::UsmOut<uint16_t>{d.p},
                                                        n, r, kernel, {})
        .wait();
    return d.host(n);
}

template <class E> static E draw(Rng &r) {
    if constexpr (std::is_same_v<E, bool>)
        return r.bit();
    else if constexpr (std::is_same_v<E, uint32_t>)
        return r.urand();
    else if constexpr (std::is_same_v<E, uint64_t>)
        return r.urand64();
    else if constexpr (std::is_same_v<E, float>)
        return r.frand();
    else
        return r.drand();
}

// Host copy of n scalar draws in a kernel, 256 consecutive draws per work item, each work item
// with its own generator placed at its first element.
template <class E>
static std::vector<host_t<E>> device_draws(sycl::queue &q, const Key &key, uint64_t pos,
                                           uint32_t K, size_t n) {
    Dev<E> d(q, n);
    constexpr size_t per = 256;
    uint64_t p0 = tandem::align_pos(pos, bits_of<E>);
    E *v = d.p;
    if (n)
        q.parallel_for(sycl::range<1>((n + per - 1) / per), [=](sycl::item<1> it) {
             size_t i = it.get_id(0);
             Rng r = Rng::from_key(key, p0 + (uint64_t)i * per * bits_of<E>, K);
             for (size_t k = i * per; k < (i + 1) * per && k < n; k++)
                 v[k] = draw<E>(r);
         }).wait();
    return d.host(n);
}

template <class A, class B> static size_t first_diff(const A &a, const B &b) {
    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == (typename A::value_type)b[i])
        i++;
    return a.size() == b.size() && i == a.size() ? SIZE_MAX : i;
}

// w bits (1 to 64) at the w-aligned stream position q, read from u32 words that start at the
// 32-aligned position base.
static uint64_t bits_at(const std::vector<uint32_t> &words, uint64_t base, uint64_t q, unsigned w) {
    size_t i = (size_t)((q - base) >> 5);
    if (w == 64)
        return words[i] | ((uint64_t)words[i + 1] << 32);
    return (words[i] >> (q & 31u)) & (0xffffffffu >> (32u - w));
}

// ---- Vectors ------------------------------------------------------------------------------

static void test_vectors(sycl::queue &q) {
    for (const auto &v : VEC_T) {
        uint32_t o[4], h[4];
        std::memcpy(o, v.o, 16);
        std::memcpy(h, v.h, 16);
        tandem::T(o, h);
        CHECK(words_equal(o, v.o_out) && words_equal(h, v.h_out));
    }
    for (const auto &v : VEC_F) {
        uint32_t o[4], h[4];
        tandem::F_keyed(VEC_KEY, v.counter, tandem::DOMAIN_STREAM, tandem::AUX_STREAM, o, h);
        CHECK(words_equal(o, v.o) && words_equal(h, v.h));
    }
    const Key key = key_of(VEC_KEY);
    for (Kernel kernel : KERNELS) {
        auto u32 = device_fill<uint32_t>(q, key, 0, VEC_K, 64, kernel);
        for (const auto &s : VEC_STREAM) {
            CHECK(std::memcmp(&u32[s.first_word], s.words, 16) == 0);
            uint64_t p = s.first_word * 32;
            uint32_t b[4];
            tandem::block(VEC_KEY, tandem::chunk_of(p, VEC_K), tandem::step_of(p, VEC_K), b);
            CHECK(words_equal(b, s.words));
        }
        auto f64 = device_fill<double>(q, key, 0, VEC_K, 32, kernel);
        for (const auto &v : VEC_F64)
            CHECK(f64[v.index] == v.value);
        auto f32 = device_fill<float>(q, key, 0, VEC_K, 32, kernel);
        for (const auto &v : VEC_F32)
            CHECK(f32[v.index] == v.value);
        auto bits = device_fill<bool>(q, key, 0, VEC_K, 129, kernel);
        for (const auto &v : VEC_BOOL)
            CHECK(bits[v.index] == (v.value != 0));
    }

    // Derived keys, computed in a kernel.
    Dev<uint32_t> k(q, 24);
    uint32_t *kp = k.p;
    q.single_task([=] {
         Rng r = Rng::from_key(key, 0, VEC_K);
         Rng kids[2];
         Rng f = r;
         f.fork(kids, 2);
         const Key ks[6] = {r.split(0).key(), r.split(1).key(), r.sub(7).key(),
                            kids[0].key(),    kids[1].key(),    Rng(VEC_SEED, 0, VEC_K).key()};
         for (int i = 0; i < 6; i++)
             for (int w = 0; w < 4; w++)
                 kp[4 * i + w] = ks[i].w[w];
     }).wait();
    auto got = k.host(24);
    CHECK(words_equal(&got[0], VEC_SPLIT0));
    CHECK(words_equal(&got[4], VEC_SPLIT1));
    CHECK(words_equal(&got[8], VEC_PURPOSE7));
    CHECK(words_equal(&got[12], VEC_FORK0));
    CHECK(words_equal(&got[20], VEC_SEED_KEY));
    // Fork child 1 is the hidden half of the same F as child 0.
    uint32_t o[4], h[4];
    tandem::F_keyed(VEC_KEY, 0, tandem::DOMAIN_FORK, 0, o, h);
    CHECK(words_equal(&got[16], h));

    const Key seed_key = Rng(VEC_SEED, 0, VEC_K).key();
    auto sf64 = device_fill<double>(q, seed_key, 0, VEC_K, 32, Kernel::Auto);
    for (const auto &v : VEC_SEED_F64)
        CHECK(sf64[v.index] == v.value);
    auto su32 = device_fill<uint32_t>(q, seed_key, 0, VEC_K, 32, Kernel::Auto);
    for (const auto &v : VEC_SEED_U32)
        CHECK(su32[v.index] == v.value);
}

// ---- Dumps --------------------------------------------------------------------------------

template <class T> static std::vector<T> slurp(const std::string &dir, const char *file) {
    std::string path = dir + "/" + file;
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::printf("FAIL cannot open %s\n", path.c_str());
        failures++;
        return {};
    }
    std::fseek(f, 0, SEEK_END);
    size_t len = (size_t)std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<T> v(len / sizeof(T));
    if (std::fread(v.data(), 1, len, f) != len)
        failures++;
    std::fclose(f);
    return v;
}

template <class E>
static void check_dump_fill(sycl::queue &q, const std::string &dir, const char *file,
                            const Key &key, uint32_t K) {
    std::vector<host_t<E>> want = slurp<host_t<E>>(dir, file);
    if (want.empty())
        return;
    for (Kernel kernel : KERNELS) {
        size_t i = first_diff(want, device_fill<E>(q, key, 0, K, want.size(), kernel));
        CHECK(i == SIZE_MAX);
        if (i != SIZE_MAX)
            std::printf("  %s: %s fill differs at %zu\n", file, name(kernel), i);
    }
}

template <class E>
static void check_dump(sycl::queue &q, const std::string &dir, const char *file, const Key &key,
                       uint32_t K) {
    check_dump_fill<E>(q, dir, file, key, K);
    std::vector<host_t<E>> want = slurp<host_t<E>>(dir, file);
    size_t i = first_diff(want, device_draws<E>(q, key, 0, K, want.size()));
    CHECK(i == SIZE_MAX);
    if (i != SIZE_MAX)
        std::printf("  %s: draws differ at %zu\n", file, i);
}

static void test_dumps(sycl::queue &q, const std::string &dir) {
    const uint32_t k1234[4] = {1, 2, 3, 4};
    const Key k = key_of(k1234), s42 = Rng(42).key();
    check_dump<uint32_t>(q, dir, "k1234_K32_u32.bin", k, 32);
    check_dump<uint64_t>(q, dir, "k1234_K32_u64.bin", k, 32);
    check_dump<uint32_t>(q, dir, "k1234_K8_u32.bin", k, 8);
    check_dump<double>(q, dir, "seed42_K32_f64.bin", s42, 32);
    check_dump<float>(q, dir, "seed42_K32_f32.bin", s42, 32);
    check_dump<bool>(q, dir, "seed42_K32_bool.bin", s42, 32);
    check_dump_fill<uint8_t>(q, dir, "seed42_K32_u8.bin", s42, 32);
    check_dump_fill<std::complex<float>>(q, dir, "seed42_K32_c32.bin", s42, 32);
    check_dump_fill<std::complex<double>>(q, dir, "seed42_K32_c64.bin", s42, 32);

    // Float16 as bit patterns and as sycl::half, whose fill writes the same bits.
    std::vector<uint16_t> f16 = slurp<uint16_t>(dir, "seed42_K32_f16bits.bin");
    for (Kernel kernel : KERNELS) {
        CHECK(first_diff(f16, device_f16_bits(q, s42, 0, 32, f16.size(), kernel)) == SIZE_MAX);
        auto h = device_fill<sycl::half>(q, s42, 0, 32, f16.size(), kernel);
        CHECK(std::memcmp(h.data(), f16.data(), 2 * f16.size()) == 0);
    }
}

// ---- Fills against draws at random keys, chunk lengths, positions, lengths, alignments ----

struct Trial {
    Key key;
    uint32_t K;
    uint64_t pos;
    size_t n, shift;
};

static std::vector<Trial> trials(uint64_t seed, int count) {
    std::mt19937_64 gen(seed);
    std::vector<Trial> out;
    for (int t = 0; t < count; t++) {
        Trial x;
        for (auto &w : x.key.w)
            w = (uint32_t)gen();
        x.K = 1u << (gen() % 9);
        x.pos = gen() % (1u << 20);
        x.n = (size_t)(gen() % (t < count * 3 / 4 ? 5000 : 200000));
        x.shift = gen() % 4;
        out.push_back(x);
    }
    // Position 0 and shift 0 put every block of the output on a 16-byte address.
    out.push_back(Trial{out.back().key, 32, 0, 5000, 0});
    return out;
}

template <class E> static void check_against_draws(sycl::queue &q, const char *label) {
    for (const Trial &t : trials(2026, 24)) {
        auto want = device_draws<E>(q, t.key, t.pos, t.K, t.n);
        for (Kernel kernel : KERNELS) {
            uint64_t end;
            auto got = device_fill<E>(q, t.key, t.pos, t.K, t.n, kernel, t.shift, &end);
            size_t i = first_diff(want, got);
            CHECK(i == SIZE_MAX);
            CHECK(end == tandem::align_pos(t.pos, bits_of<E>) + t.n * bits_of<E>);
            if (i != SIZE_MAX)
                std::printf("  %s %s fill vs draws (K=%u pos=%llu n=%zu shift=%zu) at %zu\n", label,
                            name(kernel), t.K, (unsigned long long)t.pos, t.n, t.shift, i);
        }
    }
}

// Fills of every type take the bits of the u32 stream at their own alignment, from both
// kernels.
template <class E> static void check_against_stream(sycl::queue &q, const char *label) {
    for (const Trial &t : trials(31, 12)) {
        constexpr unsigned w = bits_of<E>;
        uint64_t p0 = tandem::align_pos(t.pos, w);
        auto words = device_fill<uint32_t>(q, t.key, 0, t.K, (p0 + t.n * w) / 32 + 2,
                                           Kernel::Chunk);
        std::vector<host_t<E>> want(t.n);
        for (size_t i = 0; i < t.n; i++) {
            uint64_t raw = bits_at(words, 0, p0 + i * w, w);
            if constexpr (std::is_same_v<E, float>)
                want[i] = tandem::to_f32((uint32_t)raw);
            else if constexpr (std::is_same_v<E, double>)
                want[i] = tandem::to_f64(raw);
            else
                want[i] = (host_t<E>)raw;
        }
        for (Kernel kernel : KERNELS) {
            uint64_t end;
            auto got = device_fill<E>(q, t.key, t.pos, t.K, t.n, kernel, t.shift, &end);
            CHECK(first_diff(want, got) == SIZE_MAX);
            CHECK(end == p0 + t.n * w);
            if (first_diff(want, got) != SIZE_MAX)
                std::printf("  %s %s fill vs u32 stream (K=%u pos=%llu n=%zu shift=%zu)\n", label,
                            name(kernel), t.K, (unsigned long long)t.pos, t.n, t.shift);
        }
    }
}

// A fill split in pieces, with host draws between them, continues one stream: the pieces and
// the draws equal the u32 words of one whole fill.
static void test_split_fills(sycl::queue &q) {
    const uint32_t kw[4] = {0xdeadbeef, 7, 99, 0x12345678};
    const Key key = key_of(kw);
    for (uint32_t K : {1u, 8u, 32u}) {
        const uint64_t start = 37; // inside row 0
        const uint64_t base = tandem::align_pos(start, 32);
        auto whole = device_fill<uint32_t>(q, key, start, K, 40000, Kernel::Chunk);
        Rng r = Rng::from_key(key, start, K);
        std::vector<std::pair<uint64_t, unsigned>> at; // stream position and width of each value
        std::vector<uint64_t> got;
        bool ok = true;

        auto take = [&](auto tag, size_t n) {
            using E = decltype(tag);
            Dev<E> d(q, n);
            uint64_t p = tandem::align_pos(r.position(), bits_of<E>);
            tandem::fill(q, d.p, n, r).wait();
            auto h = d.host(n);
            for (size_t i = 0; i < n; i++) {
                at.push_back({p + i * bits_of<E>, bits_of<E> == 32 ? 32u : 64u});
                if constexpr (std::is_same_v<E, double>) {
                    uint64_t raw = bits_at(whole, base, p + i * 64, 64);
                    ok = ok && h[i] == tandem::to_f64(raw);
                    got.push_back(raw);
                } else {
                    got.push_back((uint64_t)h[i]);
                }
            }
        };
        take(uint32_t{}, 1001);
        for (int i = 0; i < 3; i++) {
            at.push_back({tandem::align_pos(r.position(), 32), 32});
            got.push_back(r.urand());
        }
        take(uint64_t{}, 5003);
        at.push_back({tandem::align_pos(r.position(), 64), 64});
        got.push_back(r.urand64());
        take(double{}, 3333);
        for (size_t i = 0; i < got.size(); i++)
            ok = ok && got[i] == bits_at(whole, base, at[i].first, at[i].second);
        CHECK(ok);
        CHECK(at.back().first + 64 <= base + 32 * whole.size());
    }
}

// Mixed widths through one in-kernel generator read the stream at the aligned positions.
static void test_mixed_draws(sycl::queue &q) {
    const uint32_t kw[4] = {9, 8, 7, 6};
    const Key key = key_of(kw);
    constexpr int rounds = 2000;
    Dev<uint64_t> d(q, 5 * rounds + 1);
    uint64_t *v = d.p;
    q.single_task([=] {
         Rng r = Rng::from_key(key, 5, 32);
         for (int i = 0; i < rounds; i++) {
             v[5 * i + 0] = r.bit();
             v[5 * i + 1] = r.urand();
             double x = r.drand();
             std::memcpy(&v[5 * i + 2], &x, 8);
             v[5 * i + 3] = r.urand64();
             float f = r.frand();
             uint32_t fb;
             std::memcpy(&fb, &f, 4);
             v[5 * i + 4] = fb;
         }
         v[5 * rounds] = r.position();
     }).wait();
    auto h = d.host(5 * rounds + 1);
    auto words = device_fill<uint32_t>(q, key, 0, 32, 5 * rounds * 6, Kernel::Auto);
    uint64_t p = 5;
    bool ok = true;
    auto next = [&](unsigned w) {
        p = tandem::align_pos(p, w);
        uint64_t x = bits_at(words, 0, p, w);
        p += w;
        return x;
    };
    for (int i = 0; i < rounds; i++) {
        ok = ok && h[5 * i + 0] == next(1);
        ok = ok && h[5 * i + 1] == next(32);
        double x = tandem::to_f64(next(64));
        ok = ok && std::memcmp(&h[5 * i + 2], &x, 8) == 0;
        ok = ok && h[5 * i + 3] == next(64);
        float f = tandem::to_f32((uint32_t)next(32));
        uint32_t fb;
        std::memcpy(&fb, &f, 4);
        ok = ok && h[5 * i + 4] == fb;
    }
    CHECK(ok);
    CHECK(h[5 * rounds] == p);

    // Random access in a kernel reads the fill that would start at the position.
    Dev<uint64_t> a(q, 4);
    uint64_t *av = a.p;
    q.single_task([=] {
         Rng r = Rng::from_key(key, 5, 32);
         double x = r.at_drand(3);
         float f = r.at_frand(7);
         uint32_t fb;
         std::memcpy(&av[0], &x, 8);
         std::memcpy(&fb, &f, 4);
         av[1] = fb;
         av[2] = r.at_urand(1000);
         av[3] = r.at_urand64(100);
     }).wait();
    auto ha = a.host(4);
    double x = tandem::to_f64(bits_at(words, 0, 64 + 192, 64));
    float f = tandem::to_f32(words[8]);
    uint32_t fb;
    std::memcpy(&fb, &f, 4);
    CHECK(std::memcmp(&ha[0], &x, 8) == 0 && ha[1] == fb);
    CHECK(ha[2] == words[1001] && ha[3] == bits_at(words, 0, 64 + 6400, 64));
}

// The bounded draws, normals and narrow draws of a generator in a kernel equal the host
// generator's: integers bit for bit, normals to the platform's log, sqrt, cos and sin.
template <class E> static bool near_normal(E got, E want) {
    if constexpr (std::is_same_v<E, double>)
        return std::abs(got - want) <= 1e-12 * std::abs(want) + 1e-14;
    else
        return std::abs(got - want) <= 16 * 0x1p-23f * std::abs(want) + 1e-6f;
}

static void test_device_draws(sycl::queue &q) {
    const Key key = Rng(7).key();
    constexpr size_t n = 3000;
    struct Out {
        uint32_t below32[n];
        uint64_t below64[n];
        int32_t signed32[n];
        int64_t signed64[n];
        double normal[n], pair[2 * n];
        float normalf[n], pairf[2 * n];
        uint64_t pos;
    };
    for (uint32_t K : {1u, 32u}) {
        Out *d = sycl::malloc_shared<Out>(1, q);
        q.single_task([=] {
             Rng r = Rng::from_key(key, 3, K);
             for (size_t i = 0; i < n; i++) {
                 d->below32[i] = r.urand(i % 2 ? 6u : 0x80000001u);
                 d->below64[i] = r.urand64(i % 2 ? 1000003ull : 0x8000000000000001ull);
                 d->signed32[i] = r.rand(-5, 5);
                 d->signed64[i] = r.rand64(-3, 1000);
                 d->normal[i] = r.normal();
                 d->normalf[i] = r.normalf();
                 auto p = r.normal2();
                 d->pair[2 * i] = p.z0, d->pair[2 * i + 1] = p.z1;
                 auto pf = r.normalf2();
                 d->pairf[2 * i] = pf.z0, d->pairf[2 * i + 1] = pf.z1;
             }
             d->pos = r.position();
         }).wait();
        Rng r = Rng::from_key(key, 3, K);
        bool ints = true, normals = true;
        for (size_t i = 0; i < n; i++) {
            ints = ints && d->below32[i] == r.urand(i % 2 ? 6u : 0x80000001u);
            ints = ints && d->below64[i] == r.urand64(i % 2 ? 1000003ull : 0x8000000000000001ull);
            ints = ints && d->signed32[i] == r.rand(-5, 5);
            ints = ints && d->signed64[i] == r.rand64(-3, 1000);
            normals = normals && near_normal(d->normal[i], r.normal());
            normals = normals && near_normal(d->normalf[i], r.normalf());
            auto p = r.normal2();
            normals = normals && near_normal(d->pair[2 * i], p.z0) &&
                      near_normal(d->pair[2 * i + 1], p.z1);
            auto pf = r.normalf2();
            normals = normals && near_normal(d->pairf[2 * i], pf.z0) &&
                      near_normal(d->pairf[2 * i + 1], pf.z1);
        }
        CHECK(ints);
        CHECK(normals);
        CHECK(d->pos == r.position());
        sycl::free(d, q);
    }
}

// ---- Value types ---------------------------------------------------------------------------

// A signed fill holds the two's complement of the unsigned fill of the same width.
template <class S, class U> static void check_signed(sycl::queue &q) {
    for (const Trial &t : trials(32, 6)) {
        auto u = device_fill<U>(q, t.key, t.pos, t.K, t.n, Kernel::Auto, t.shift);
        auto s = device_fill<S>(q, t.key, t.pos, t.K, t.n, Kernel::Auto, t.shift);
        CHECK(std::memcmp(u.data(), s.data(), t.n * sizeof(S)) == 0);
    }
}

// The binary16 bits decode to (raw >> 5) * 2^-11, checked by decoding the fields by hand.
static void test_f16(sycl::queue &q) {
    for (const Trial &t : trials(33, 8)) {
        auto raw = device_fill<uint16_t>(q, t.key, t.pos, t.K, t.n, Kernel::Auto);
        for (Kernel kernel : KERNELS) {
            auto bits = device_f16_bits(q, t.key, t.pos, t.K, t.n, kernel);
            bool ok = true;
            for (size_t i = 0; i < t.n; i++) {
                unsigned e = bits[i] >> 10 & 31u, m = bits[i] & 1023u;
                double v = bits[i] == 0 ? 0.0 : std::ldexp(1.0 + m / 1024.0, (int)e - 15);
                ok = ok && !(bits[i] >> 15) && e != 31 && v == (raw[i] >> 5) * 0x1p-11;
            }
            CHECK(ok);
        }
    }
}

// A complex value takes two draws, the real and then the imaginary component.
template <class T> static void check_complex(sycl::queue &q) {
    for (const Trial &t : trials(34, 8)) {
        auto parts = device_fill<T>(q, t.key, t.pos, t.K, 2 * t.n, Kernel::Auto);
        for (Kernel kernel : KERNELS) {
            uint64_t end;
            auto c =
                device_fill<std::complex<T>>(q, t.key, t.pos, t.K, t.n, kernel, t.shift, &end);
            CHECK(std::memcmp(c.data(), parts.data(), t.n * sizeof(std::complex<T>)) == 0);
            CHECK(end == tandem::align_pos(t.pos, 8 * sizeof(T)) + 16 * sizeof(T) * t.n);
        }
    }
}

// Buffers of rank 1 and 2 fill in linear order with the values and positions of USM fills.
static void test_buffers(sycl::queue &q) {
    const Key key = Rng(5).key();
    constexpr size_t a = 37, b = 29, n = a * b;
    Rng u = Rng::from_key(key, 3, 32), g = u;
    Dev<double> d(q, n);
    Dev<uint8_t> d8(q, n);
    Dev<uint32_t> db(q, n);
    Dev<float> dn(q, n);
    Dev<uint16_t> df(q, n / 2);
    tandem::fill(q, d.p, n, u).wait();
    tandem::fill(q, d8.p, n, u).wait();
    tandem::fill_below(q, db.p, n, u, 1000u).wait();
    tandem::fill_normal(q, dn.p, n, u).wait();
    tandem::fill_f16_bits(q, df.p, n / 2, u).wait();

    std::vector<double> h(n);
    std::vector<uint8_t> h8(n);
    std::vector<uint32_t> hb(n);
    std::vector<float> hn(n);
    std::vector<uint16_t> hf(n / 2);
    {
        sycl::buffer<double, 2> b2(h.data(), sycl::range<2>(a, b));
        sycl::buffer<uint8_t, 1> b8(h8.data(), sycl::range<1>(n));
        sycl::buffer<uint32_t, 1> bb(hb.data(), sycl::range<1>(n));
        sycl::buffer<float, 2> bn(hn.data(), sycl::range<2>(a, b));
        sycl::buffer<uint16_t, 1> bf(hf.data(), sycl::range<1>(n / 2));
        tandem::fill(q, b2, g);
        tandem::fill(q, b8, g);
        tandem::fill_below(q, bb, g, 1000u);
        tandem::fill_normal(q, bn, g);
        tandem::fill_f16_bits(q, bf, g);
    }
    CHECK(first_diff(h, d.host(n)) == SIZE_MAX);
    CHECK(first_diff(h8, d8.host(n)) == SIZE_MAX);
    CHECK(first_diff(hb, db.host(n)) == SIZE_MAX);
    CHECK(first_diff(hn, dn.host(n)) == SIZE_MAX);
    CHECK(first_diff(hf, df.host(n / 2)) == SIZE_MAX);
    CHECK(g.position() == u.position());
}

// ---- Bounded and normal fills --------------------------------------------------------------

// The contract in core.hpp, written out on host generators: element e takes draw e of the
// fill, and a rejected draw retries on the draws of split(e) of sub(PURPOSE_BELOW) of the
// fill's generator at position 0.
static uint32_t ref_below32(const Key &key, uint32_t K, uint32_t u, uint32_t range, uint64_t e) {
    uint64_t m = (uint64_t)u * range;
    if ((uint32_t)m < range) {
        uint32_t t = (0u - range) % range;
        if ((uint32_t)m < t) {
            Rng f = Rng::from_key(key, 0, K).sub(tandem::PURPOSE_BELOW32).split(e);
            do
                m = (uint64_t)f.urand() * range;
            while ((uint32_t)m < t);
        }
    }
    return (uint32_t)(m >> 32);
}

static uint64_t ref_below64(const Key &key, uint32_t K, uint64_t x, uint64_t range, uint64_t e) {
    unsigned __int128 m = (unsigned __int128)x * range;
    if ((uint64_t)m < range) {
        uint64_t t = (0u - range) % range;
        if ((uint64_t)m < t) {
            Rng f = Rng::from_key(key, 0, K).sub(tandem::PURPOSE_BELOW64).split(e);
            do
                m = (unsigned __int128)f.urand64() * range;
            while ((uint64_t)m < t);
        }
    }
    return (uint64_t)(m >> 64);
}

template <class E> static E ref_below(const Key &key, uint32_t K, E draw, E range, uint64_t e) {
    if constexpr (sizeof(E) == 4)
        return ref_below32(key, K, draw, range, e);
    else
        return ref_below64(key, K, draw, range, e);
}

template <class E>
static std::vector<E> device_below(sycl::queue &q, const Key &key, uint64_t pos, uint32_t K,
                                   size_t n, E range, Kernel kernel, size_t shift,
                                   uint64_t *end) {
    using Kind = typename tandem::detail::below<E>::kind;
    Dev<E> d(q, n + 4);
    Rng r = Rng::from_key(key, pos, K);
    tandem::detail::fill_kind<Kind>(q, tandem::detail::UsmOut<E>{d.p + shift}, n, r, kernel, {},
                                    range)
        .wait();
    *end = r.position();
    return d.host(n, shift);
}

// Ranges 0 and 1, small ranges, 2^31 + 1 and the maximum, which reject often.
template <class E> static void check_below(sycl::queue &q, const char *label) {
    const E ranges[] = {0, 1, 3, 1000, (E)1 << (8 * sizeof(E) - 1) | 1u, (E) ~(E)0};
    for (const Trial &t : trials(41, 12)) {
        auto draws = device_fill<E>(q, t.key, t.pos, t.K, t.n, Kernel::Chunk);
        for (E range : ranges) {
            uint64_t p0 = tandem::align_pos(t.pos, 8 * sizeof(E));
            std::vector<E> want(t.n);
            for (size_t i = 0; i < t.n; i++)
                want[i] = ref_below<E>(t.key, t.K, draws[i], range, i);
            for (Kernel kernel : KERNELS) {
                uint64_t end;
                auto got = device_below<E>(q, t.key, t.pos, t.K, t.n, range, kernel, t.shift,
                                           &end);
                CHECK(first_diff(want, got) == SIZE_MAX);
                CHECK(end == p0 + t.n * 8 * sizeof(E));
                if (first_diff(want, got) != SIZE_MAX)
                    std::printf("  %s %s below(%llu) (K=%u pos=%llu n=%zu shift=%zu) at %zu\n",
                                label, name(kernel), (unsigned long long)range, t.K,
                                (unsigned long long)t.pos, t.n, t.shift, first_diff(want, got));
            }
        }
    }
}

// Without a rejection a bounded fill equals the sequential urand(range) calls, and fixtures from
// tandem-cuda pin the fallback stream: 41 of the 2^31 + 1 elements and 34 of the 2^63 + 1
// elements reject.
static void test_below(sycl::queue &q) {
    check_below<uint32_t>(q, "u32");
    check_below<uint64_t>(q, "u64");

    const Key key = Rng(11).key();
    uint64_t end0;
    auto got = device_below<uint32_t>(q, key, 5, 32, 5000, 1000u, Kernel::Auto, 0, &end0);
    Rng r = Rng::from_key(key, 5, 32);
    bool same = true;
    for (uint32_t v : got)
        same = same && v == r.urand(1000u);
    CHECK(same);

    CHECK(words_equal(CROSS_FILL_KEY, Rng(42).key().w));
    const Key k42 = Rng(42).key();
    for (Kernel kernel : KERNELS) {
        uint64_t end;
        for (const auto &f : CROSS_BELOW32) {
            auto v = device_below<uint32_t>(q, k42, 0, 32, 64, f.range, kernel, 0, &end);
            CHECK(std::memcmp(v.data(), f.out, sizeof f.out) == 0 && end == 64 * 32);
        }
        for (const auto &f : CROSS_BELOW64) {
            auto v = device_below<uint64_t>(q, k42, 0, 32, 64, f.range, kernel, 0, &end);
            CHECK(std::memcmp(v.data(), f.out, sizeof f.out) == 0 && end == 64 * 64);
        }
    }
}

template <class E>
static std::vector<E> device_normal(sycl::queue &q, const Key &key, uint64_t pos, uint32_t K,
                                    size_t n, size_t shift, uint64_t *end) {
    Dev<E> d(q, n + 4);
    Rng r = Rng::from_key(key, pos, K);
    tandem::fill_normal(q, d.p + shift, n, r).wait();
    *end = r.position();
    return d.host(n, shift);
}

// The flattened normal2 calls: an odd count drops the last sin half and still consumes both
// draws.
template <class E> static std::vector<E> sequential_normals(Rng &r, size_t n) {
    std::vector<E> z(n);
    for (size_t i = 0; i < n; i += 2) {
        auto pair = [&] {
            if constexpr (std::is_same_v<E, double>)
                return r.normal2();
            else
                return r.normalf2();
        }();
        z[i] = pair.z0;
        if (i + 1 < n)
            z[i + 1] = pair.z1;
    }
    return z;
}

template <class E> static void check_normal(sycl::queue &q, const char *label) {
    for (const Trial &t : trials(42, 24)) {
        Rng r = Rng::from_key(t.key, t.pos, t.K);
        std::vector<E> want = sequential_normals<E>(r, t.n);
        uint64_t end;
        auto got = device_normal<E>(q, t.key, t.pos, t.K, t.n, t.shift, &end);
        bool ok = true;
        for (size_t i = 0; i < t.n; i++)
            ok = ok && near_normal(got[i], want[i]);
        CHECK(ok);
        CHECK(end == r.position());
        if (!ok)
            std::printf("  %s normal (K=%u pos=%llu n=%zu shift=%zu)\n", label, t.K,
                        (unsigned long long)t.pos, t.n, t.shift);
    }
}

// Fixtures from tandem-cuda at positions that put the first pair at an even and an odd draw,
// with an odd count.
static void test_normal(sycl::queue &q) {
    check_normal<double>(q, "f64");
    check_normal<float>(q, "f32");

    const Key k42 = Rng(42).key();
    for (const auto &f : CROSS_NORMAL64) {
        uint64_t end;
        auto v = device_normal<double>(q, k42, f.pos, 32, f.n, 0, &end);
        bool ok = true;
        for (unsigned i = 0; i < f.n; i++)
            ok = ok && near_normal(v[i], f.out[i]);
        CHECK(ok);
        CHECK(end == tandem::align_pos(f.pos, 64) + 128 * ((f.n + 1) / 2));
    }
    for (const auto &f : CROSS_NORMAL32) {
        uint64_t end;
        auto v = device_normal<float>(q, k42, f.pos, 32, f.n, 0, &end);
        bool ok = true;
        for (unsigned i = 0; i < f.n; i++)
            ok = ok && near_normal(v[i], f.out[i]);
        CHECK(ok);
        CHECK(end == tandem::align_pos(f.pos, 32) + 64 * ((f.n + 1) / 2));
    }
}

// Positions after a fill follow tandem-cuda: an empty fill of a type aligns the position to its
// width, and an empty bounded or normal fill consumes no draws and leaves it alone.
static void test_positions(sycl::queue &q) {
    const Key key = Rng(3).key();
    Dev<uint64_t> d(q, 4);
    for (uint64_t pos : {0ull, 1ull, 33ull, 64ull, 65ull, 1001ull}) {
        Rng r = Rng::from_key(key, pos, 32);
        tandem::fill(q, reinterpret_cast<uint32_t *>(d.p), 0, r).wait();
        CHECK(r.position() == tandem::align_pos(pos, 32));
        r.set_position(pos);
        tandem::fill(q, reinterpret_cast<std::complex<double> *>(d.p), 0, r).wait();
        CHECK(r.position() == tandem::align_pos(pos, 64));
        r.set_position(pos);
        tandem::fill(q, reinterpret_cast<bool *>(d.p), 0, r).wait();
        CHECK(r.position() == pos);
        r.set_position(pos);
        tandem::fill_normal(q, reinterpret_cast<double *>(d.p), 0, r).wait();
        tandem::fill_normal(q, reinterpret_cast<float *>(d.p), 0, r).wait();
        tandem::fill_below(q, reinterpret_cast<uint32_t *>(d.p), 0, r, 6u).wait();
        tandem::fill_below(q, d.p, 0, r, 6u).wait();
        CHECK(r.position() == pos);
        // One element moves the position past one aligned draw, or past a pair for normals.
        tandem::fill_below(q, d.p, 1, r, 6u).wait();
        CHECK(r.position() == tandem::align_pos(pos, 64) + 64);
        r.set_position(pos);
        tandem::fill_normal(q, reinterpret_cast<float *>(d.p), 1, r).wait();
        CHECK(r.position() == tandem::align_pos(pos, 32) + 64);
    }
    // A fill that would run past position 2^64 throws before it moves the position.
    Rng r = Rng::from_key(key, ~0ull - 100, 32);
    bool threw = false;
    try {
        tandem::fill(q, d.p, 4, r);
    } catch (const std::overflow_error &) {
        threw = true;
    }
    CHECK(threw && r.position() == ~0ull - 100);
}

int main(int argc, char **argv) {
    std::string dir = argc > 1 ? argv[argc - 1] : "tests/data";
    sycl::queue q;
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    test_vectors(q);
    test_dumps(q, dir);
    check_against_draws<uint32_t>(q, "u32");
    check_against_draws<uint64_t>(q, "u64");
    check_against_draws<float>(q, "f32");
    check_against_draws<double>(q, "f64");
    check_against_draws<bool>(q, "bool");
    check_against_stream<bool>(q, "bool");
    check_against_stream<uint8_t>(q, "u8");
    check_against_stream<uint16_t>(q, "u16");
    check_against_stream<uint32_t>(q, "u32");
    check_against_stream<uint64_t>(q, "u64");
    check_against_stream<float>(q, "f32");
    check_against_stream<double>(q, "f64");
    test_split_fills(q);
    test_mixed_draws(q);
    test_device_draws(q);
    check_signed<int8_t, uint8_t>(q);
    check_signed<int16_t, uint16_t>(q);
    check_signed<int32_t, uint32_t>(q);
    check_signed<int64_t, uint64_t>(q);
    test_f16(q);
    check_complex<float>(q);
    check_complex<double>(q);
    test_buffers(q);
    test_below(q);
    test_normal(q);
    test_positions(q);
    if (failures) {
        std::printf("%ld of %ld checks failed\n", failures, checks);
        return 1;
    }
    std::printf("ok: %ld checks\n", checks);
    return 0;
}
