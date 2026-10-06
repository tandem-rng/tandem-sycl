// Spec vectors, the spec's conformance files, fills against in-kernel draws and the u32 stream,
// split fills, derived keys, bounded, normal and exponential fills, buffers, and position
// advancement, on the default SYCL device.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>

#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <tandem/sycl.hpp>

#include "conformance.hpp"
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

// ---- Stream hashes of hashes.json ---------------------------------------------------------

static Key json_key(const Json &words) {
    Key k;
    for (int w = 0; w < 4; w++)
        k.w[w] = (uint32_t)words.items[w].hex();
    return k;
}

template <class T> static std::string sha256_of(const std::vector<T> &v) {
    return sha256_hex(v.data(), v.size() * sizeof(T));
}

// The fills of both kernels, and the in-kernel scalar draws of the types that have one, hash to
// the stream's SHA-256. Float16 also as bit patterns, which sycl::half fills write too.
template <class E> static void check_stream(sycl::queue &q, const Json &s) {
    const Key key = json_key(s["key"]);
    const uint32_t K = (uint32_t)s["K"].u64();
    const uint64_t start = s["start"].u64();
    const size_t n = s["n"].u64();
    const std::string &want = s["sha256"].text;
    for (Kernel kernel : KERNELS) {
        bool ok = sha256_of(device_fill<E>(q, key, start, K, n, kernel)) == want;
        if constexpr (std::is_same_v<E, sycl::half>)
            ok = ok && sha256_of(device_f16_bits(q, key, start, K, n, kernel)) == want;
        CHECK(ok);
        if (!ok)
            std::printf("  %s: %s fill hash differs\n", s["file"].text.c_str(), name(kernel));
    }
    if constexpr (std::is_same_v<E, bool> || std::is_same_v<E, uint32_t> ||
                  std::is_same_v<E, uint64_t> || std::is_same_v<E, float> ||
                  std::is_same_v<E, double>)
        CHECK(sha256_of(device_draws<E>(q, key, start, K, n)) == want);
}

// UInt128 and Char have no fill here.
static void test_streams(sycl::queue &q, const Json &hashes) {
    for (const Json &s : hashes["streams"].items) {
        const std::string &type = s["type"].text;
        if (type == "UInt32")
            check_stream<uint32_t>(q, s);
        else if (type == "UInt64")
            check_stream<uint64_t>(q, s);
        else if (type == "UInt8")
            check_stream<uint8_t>(q, s);
        else if (type == "Bool")
            check_stream<bool>(q, s);
        else if (type == "Float32")
            check_stream<float>(q, s);
        else if (type == "Float64")
            check_stream<double>(q, s);
        else if (type == "Float16")
            check_stream<sycl::half>(q, s);
        else if (type == "ComplexF32")
            check_stream<std::complex<float>>(q, s);
        else if (type == "ComplexF64")
            check_stream<std::complex<double>>(q, s);
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

// The bounded draws, normals, exponentials and narrow draws of a generator in a kernel equal
// the host generator's: integers, double normals and exponentials bit for bit, float normals to
// the tolerance of Appendix A.
static bool near_normal(float got, float want) {
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
        double normal[n], pair[2 * n], expo[n];
        float normalf[n], pairf[2 * n], expof[n];
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
                 d->expo[i] = r.exponential();
                 d->expof[i] = r.exponentialf();
             }
             d->pos = r.position();
         }).wait();
        Rng r = Rng::from_key(key, 3, K);
        bool ints = true, normals = true, exps = true;
        for (size_t i = 0; i < n; i++) {
            ints = ints && d->below32[i] == r.urand(i % 2 ? 6u : 0x80000001u);
            ints = ints && d->below64[i] == r.urand64(i % 2 ? 1000003ull : 0x8000000000000001ull);
            ints = ints && d->signed32[i] == r.rand(-5, 5);
            ints = ints && d->signed64[i] == r.rand64(-3, 1000);
            double z = r.normal();
            normals = normals && std::memcmp(&d->normal[i], &z, 8) == 0;
            normals = normals && near_normal(d->normalf[i], r.normalf());
            auto p = r.normal2();
            normals = normals && std::memcmp(&d->pair[2 * i], &p.z0, 8) == 0 &&
                      std::memcmp(&d->pair[2 * i + 1], &p.z1, 8) == 0;
            auto pf = r.normalf2();
            normals = normals && near_normal(d->pairf[2 * i], pf.z0) &&
                      near_normal(d->pairf[2 * i + 1], pf.z1);
            double e = r.exponential();
            float ef = r.exponentialf();
            exps = exps && std::memcmp(&d->expo[i], &e, 8) == 0 &&
                   std::memcmp(&d->expof[i], &ef, 4) == 0;
        }
        CHECK(ints);
        CHECK(normals);
        CHECK(exps);
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
    Dev<double> dz(q, n);
    Dev<uint16_t> df(q, n / 2);
    Dev<double> de(q, n);
    tandem::fill(q, d.p, n, u).wait();
    tandem::fill(q, d8.p, n, u).wait();
    tandem::fill_below(q, db.p, n, u, 1000u).wait();
    tandem::fill_normal(q, dn.p, n, u).wait();
    tandem::fill_normal(q, dz.p, n, u).wait();
    tandem::fill_f16_bits(q, df.p, n / 2, u).wait();
    tandem::fill_exponential(q, de.p, n, u).wait();

    std::vector<double> h(n);
    std::vector<uint8_t> h8(n);
    std::vector<uint32_t> hb(n);
    std::vector<float> hn(n);
    std::vector<double> hz(n);
    std::vector<uint16_t> hf(n / 2);
    std::vector<double> he(n);
    {
        sycl::buffer<double, 2> b2(h.data(), sycl::range<2>(a, b));
        sycl::buffer<uint8_t, 1> b8(h8.data(), sycl::range<1>(n));
        sycl::buffer<uint32_t, 1> bb(hb.data(), sycl::range<1>(n));
        sycl::buffer<float, 2> bn(hn.data(), sycl::range<2>(a, b));
        sycl::buffer<double, 1> bz(hz.data(), sycl::range<1>(n));
        sycl::buffer<uint16_t, 1> bf(hf.data(), sycl::range<1>(n / 2));
        sycl::buffer<double, 2> be(he.data(), sycl::range<2>(a, b));
        tandem::fill(q, b2, g);
        tandem::fill(q, b8, g);
        tandem::fill_below(q, bb, g, 1000u);
        tandem::fill_normal(q, bn, g);
        tandem::fill_normal(q, bz, g);
        tandem::fill_f16_bits(q, bf, g);
        tandem::fill_exponential(q, be, g);
    }
    CHECK(first_diff(h, d.host(n)) == SIZE_MAX);
    CHECK(first_diff(h8, d8.host(n)) == SIZE_MAX);
    CHECK(first_diff(hb, db.host(n)) == SIZE_MAX);
    CHECK(first_diff(hn, dn.host(n)) == SIZE_MAX);
    CHECK(first_diff(hz, dz.host(n)) == SIZE_MAX);
    CHECK(first_diff(hf, df.host(n / 2)) == SIZE_MAX);
    CHECK(first_diff(he, de.host(n)) == SIZE_MAX);
    CHECK(g.position() == u.position());
}

// ---- Bounded and normal fills --------------------------------------------------------------

// Appendix A of the specification, written out on host generators: element i takes draw i of
// the fill, and a rejected draw retries on the draws of split(g) of sub(PURPOSE_BELOW) of the
// fill's generator at position 0, g = aligned start / width + i.
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
                want[i] = ref_below<E>(t.key, t.K, draws[i], range, p0 / (8 * sizeof(E)) + i);
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

// Without a rejection a bounded fill equals the sequential urand(range) calls.
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
}

// ---- Float64 normal fills ------------------------------------------------------------------

static std::vector<double> device_normal64(sycl::queue &q, const Key &key, uint64_t pos,
                                           uint32_t K, size_t n, Kernel kernel, size_t shift,
                                           uint64_t *end) {
    Dev<double> d(q, n + 4);
    Rng r = Rng::from_key(key, pos, K);
    tandem::detail::fill_normal_kind<double>(q, tandem::detail::UsmOut<double>{d.p + shift}, n,
                                             r, {}, kernel)
        .wait();
    *end = r.position();
    return d.host(n, shift);
}

static bool same_bits(const std::vector<double> &a, const std::vector<double> &b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 8) == 0;
}

// Both kernels equal the host's normal() calls bit for bit, end position included, at random
// keys, chunk lengths, positions, lengths and alignments. The trials hold misses of the fast
// path, and the tail beyond R among them.
static void check_normal64(sycl::queue &q) {
    size_t misses = 0, tails = 0;
    for (const Trial &t : trials(1618, 16)) {
        Rng r = Rng::from_key(t.key, t.pos, t.K), u = r;
        std::vector<double> want(t.n);
        for (double &x : want)
            x = r.normal();
        for (size_t i = 0; i < t.n; i++) {
            uint64_t d = u.urand64();
            bool hit;
            tandem::normal_f64_fast(d, hit);
            misses += !hit;
            tails += !hit && (d & 1023u) == 0;
        }
        for (Kernel kernel : KERNELS) {
            uint64_t end;
            auto got = device_normal64(q, t.key, t.pos, t.K, t.n, kernel, t.shift, &end);
            CHECK(same_bits(got, want) && end == r.position());
            if (!same_bits(got, want))
                std::printf("  f64 %s normal (K=%u pos=%llu n=%zu shift=%zu)\n", name(kernel),
                            t.K, (unsigned long long)t.pos, t.n, t.shift);
        }
    }
    CHECK(misses > 0 && tails > 0);
}

// A fill cut at any element equals the whole fill: at an odd element, at the first missed
// element and just after it, at a start whose global draw index differs from the element index.
static void check_normal64_cut(sycl::queue &q) {
    constexpr size_t n = 3000;
    const Key key = Rng(5, 6).key();
    const uint64_t start = 77;
    Rng u = Rng::from_key(key, start, 32);
    size_t m = 0;
    for (bool hit = true; hit; m++)
        tandem::normal_f64_fast(u.urand64(), hit);
    m--;
    for (size_t cut : {(size_t)1, (size_t)1001, m, m + 1})
        for (Kernel kernel : KERNELS) {
            uint64_t end_whole, end_a, end_b;
            auto whole = device_normal64(q, key, start, 32, n, kernel, 0, &end_whole);
            auto a = device_normal64(q, key, start, 32, cut, kernel, 0, &end_a);
            auto b = device_normal64(q, key, end_a, 32, n - cut, kernel, 0, &end_b);
            a.insert(a.end(), b.begin(), b.end());
            CHECK(same_bits(whole, a) && end_whole == end_b);
        }
}

// A miss list too short for the fill's misses makes the second kernel walk the whole fill,
// with outputs on and 8 bytes off the stream's blocks.
static void check_normal64_overflow(sycl::queue &q) {
    constexpr size_t n = 100000;
    const Key key = Rng(5, 6).key();
    Rng r = Rng::from_key(key, 77, 32);
    std::vector<double> want(n);
    for (double &x : want)
        x = r.normal();
    for (uint64_t cap : {0ull, 1ull})
        for (size_t shift : {0, 1}) {
            Dev<double> d(q, n + 1);
            Rng g = Rng::from_key(key, 77, 32);
            tandem::detail::Span s;
            tandem::detail::plan_span(g, n, 64, 64, s);
            tandem::detail::set_rows(s, s.p0 >> 7, (s.p1 - 1) >> 7);
            tandem::detail::fill_normal64_list(q, s, tandem::detail::UsmOut<double>{d.p + shift},
                                               {}, cap)
                .wait();
            CHECK(same_bits(d.host(n, shift), want) && g.position() == r.position());
        }
}

// The table pass writes octets from the fill's first draw D0: every D0 % 16, which sets the
// block's lane and the draw within it, then K = 1, whose every octet takes the next group's first
// row, and outputs on and 8 bytes off 16-byte addresses, with each way of passing values.
static void check_normal64_octets(sycl::queue &q) {
    using tandem::detail::Exchange;
    constexpr size_t n = 70001;
    const Key key = Rng(8, 9).key();
    std::vector<Exchange> exchanges = {Exchange::Local};
    if (tandem::detail::subgroups_hold_groups(q.get_device()))
        exchanges.push_back(Exchange::Shuffle);
    for (uint64_t D0 : {0ull, 1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 8ull, 9ull, 10ull, 11ull,
                        12ull, 13ull, 14ull, 15ull, (1ull << 30) + 3})
        for (uint32_t K : {1u, 32u}) {
            Rng r = Rng::from_key(key, 64 * D0, K);
            std::vector<double> want(n);
            for (double &x : want)
                x = r.normal();
            for (size_t shift : {0, 1})
                for (Exchange ex : exchanges) {
                    Dev<double> d(q, n + 1);
                    Rng g = Rng::from_key(key, 64 * D0, K);
                    tandem::detail::Span s;
                    tandem::detail::plan_span(g, n, 64, 64, s);
                    tandem::detail::set_rows(s, s.p0 >> 7, (s.p1 - 1) >> 7);
                    tandem::detail::fill_normal64_list(
                        q, s, tandem::detail::UsmOut<double>{d.p + shift}, {}, ~0ull, ex)
                        .wait();
                    bool ok = same_bits(d.host(n, shift), want);
                    CHECK(ok);
                    if (!ok)
                        std::printf("  f64 normal octets (D0=%llu K=%u shift=%zu %s)\n",
                                    (unsigned long long)D0, K, shift,
                                    ex == Exchange::Local ? "local" : "shuffle");
                }
        }
}

static void test_normal64(sycl::queue &q) {
    check_normal64(q);
    check_normal64_cut(q);
    check_normal64_overflow(q);
    check_normal64_octets(q);
}

// ---- Float32 normal fills ------------------------------------------------------------------

// The float normal fill's kernels: the chunk kernel, and the group kernel passing blocks
// through local memory and, where sub-groups hold whole groups of eight lanes, by shuffles.
using tandem::detail::Exchange;
struct NormalKernel {
    Kernel kernel;
    Exchange exchange;
    const char *name;
};
static std::vector<NormalKernel> normal_kernels(sycl::queue &q) {
    std::vector<NormalKernel> v = {{Kernel::Chunk, Exchange::Auto, "chunk"},
                                   {Kernel::Tile, Exchange::Local, "group/local"}};
    if (tandem::detail::subgroups_hold_groups(q.get_device()))
        v.push_back({Kernel::Tile, Exchange::Shuffle, "group/shuffle"});
    return v;
}

static std::vector<float> device_normal(sycl::queue &q, const Key &key, uint64_t pos, uint32_t K,
                                        size_t n, size_t shift, uint64_t *end,
                                        NormalKernel nk) {
    Dev<float> d(q, n + 4);
    Rng r = Rng::from_key(key, pos, K);
    tandem::detail::fill_normal_f32_kind(q, tandem::detail::UsmOut<float>{d.p + shift}, n, r, {},
                                         nk.kernel, nk.exchange)
        .wait();
    *end = r.position();
    return d.host(n, shift);
}

// The flattened normalf2 calls: an odd count drops the last sin half and still consumes both
// draws.
static std::vector<float> sequential_normals(Rng &r, size_t n) {
    std::vector<float> z(n);
    for (size_t i = 0; i < n; i += 2) {
        auto pair = r.normalf2();
        z[i] = pair.z0;
        if (i + 1 < n)
            z[i + 1] = pair.z1;
    }
    return z;
}

static void check_normal_at(sycl::queue &q, const Trial &t, NormalKernel kernel) {
    Rng r = Rng::from_key(t.key, t.pos, t.K);
    std::vector<float> want = sequential_normals(r, t.n);
    uint64_t end;
    auto got = device_normal(q, t.key, t.pos, t.K, t.n, t.shift, &end, kernel);
    size_t bad = SIZE_MAX;
    for (size_t i = 0; i < t.n && bad == SIZE_MAX; i++)
        if (!near_normal(got[i], want[i]))
            bad = i;
    CHECK(bad == SIZE_MAX);
    CHECK(end == r.position());
    if (bad != SIZE_MAX)
        std::printf("  f32 %s normal (K=%u pos=%llu n=%zu shift=%zu) at %zu\n", kernel.name, t.K,
                    (unsigned long long)t.pos, t.n, t.shift, bad);
}

// The group kernel cuts the fill into units of four words from its first word S and writes
// them in octets aligned to the output, so the starts cover S % 4 (the word in a block) and
// S / 4 % 8 (the lane of the first block) in several combinations, with group and work group
// boundaries and every K kind.
static void test_normal32(sycl::queue &q) {
    for (const Trial &t : trials(42, 24))
        for (NormalKernel kernel : normal_kernels(q))
            check_normal_at(q, t, kernel);
    const Key key = Rng(9).key();
    for (uint32_t K : {1u, 2u, 8u, 16u, 64u})
        for (uint64_t pos : {0ull, 32ull, 64ull, 96ull, 128ull, 416ull, 960ull, 8160ull, 8191ull})
            for (size_t n : {(size_t)1, (size_t)2, (size_t)3, (size_t)70001, (size_t)70002})
                for (NormalKernel kernel : normal_kernels(q))
                    check_normal_at(q, Trial{key, K, pos, n, n % 2 ? 1u : 0u}, kernel);

}

// ---- Exponential fills ---------------------------------------------------------------------

template <class E>
static std::vector<E> device_exponential(sycl::queue &q, const Key &key, uint64_t pos, uint32_t K,
                                         size_t n, Kernel kernel, size_t shift, uint64_t *end) {
    Dev<E> d(q, n + 4);
    Rng r = Rng::from_key(key, pos, K);
    tandem::detail::fill_kind<typename tandem::detail::exponential<E>::kind>(
        q, tandem::detail::UsmOut<E>{d.p + shift}, n, r, kernel, {})
        .wait();
    *end = r.position();
    return d.host(n, shift);
}

template <class E> static E host_exponential(Rng &r) {
    if constexpr (std::is_same_v<E, double>)
        return r.exponential();
    else
        return r.exponentialf();
}

// Both kernels equal the host's exponential() calls bit for bit at random keys, chunk lengths,
// positions, lengths and alignments.
template <class E> static void check_exponential(sycl::queue &q, const char *label) {
    for (const Trial &t : trials(271, 12)) {
        Rng r = Rng::from_key(t.key, t.pos, t.K);
        std::vector<E> want(t.n);
        for (E &x : want)
            x = host_exponential<E>(r);
        for (Kernel kernel : KERNELS) {
            uint64_t end;
            auto got = device_exponential<E>(q, t.key, t.pos, t.K, t.n, kernel, t.shift, &end);
            bool same = std::memcmp(got.data(), want.data(), t.n * sizeof(E)) == 0;
            CHECK(same && end == r.position());
            if (!same)
                std::printf("  %s %s exponential (K=%u pos=%llu n=%zu shift=%zu)\n", label,
                            name(kernel), t.K, (unsigned long long)t.pos, t.n, t.shift);
        }
    }
}

// Moments to the fourth order and a Kolmogorov-Smirnov statistic of the Exp(1) law on 10^7
// samples. The statistic is taken at the edges of 2^16 equal bins of the CDF, which never
// exceeds the supremum, so the usual critical value is conservative.
template <class E> static void check_exp1(sycl::queue &q, const char *label) {
    constexpr size_t n = 10000000, bins = 1 << 16;
    Dev<E> d(q, n);
    Rng r(1);
    tandem::fill_exponential(q, d.p, n, r).wait();
    std::vector<uint32_t> hist(bins);
    double sums[4] = {0, 0, 0, 0};
    for (E e : d.host(n)) {
        double x = e, cdf = -std::expm1(-x);
        hist[std::min((size_t)(cdf * bins), bins - 1)]++;
        for (double k = 0, p = x; k < 4; k++, p *= x)
            sums[(int)k] += p;
    }
    // E X^k = k!, and the variance of X^k is (2k)! - (k!)^2.
    const double fact[4] = {1, 2, 6, 24}, var[4] = {1, 20, 684, 39744};
    for (int k = 0; k < 4; k++) {
        double m = sums[k] / n;
        CHECK(std::abs(m - fact[k]) < 5 * std::sqrt(var[k] / n));
        if (std::abs(m - fact[k]) >= 5 * std::sqrt(var[k] / n))
            std::printf("  %s exponential moment %d: %g\n", label, k + 1, m);
    }
    double cum = 0, dmax = 0;
    for (size_t i = 0; i < bins; i++) {
        cum += hist[i];
        dmax = std::max(dmax, std::abs(cum / n - (double)(i + 1) / bins));
    }
    // P(sqrt(n) D > 1.95) is 0.001.
    CHECK(dmax * std::sqrt((double)n) < 1.95);
}

static void test_exponential(sycl::queue &q) {
    check_exponential<double>(q, "f64");
    check_exponential<float>(q, "f32");
    check_exp1<double>(q, "f64");
    check_exp1<float>(q, "f32");
}

// Positions after a fill follow tandem-cuda: an empty fill of a type or of double normals aligns
// the position to its width, and an empty bounded, float normal or exponential fill consumes no
// draws and leaves it alone.
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
        CHECK(r.position() == tandem::align_pos(pos, 64));
        r.set_position(pos);
        tandem::fill_normal(q, reinterpret_cast<float *>(d.p), 0, r).wait();
        tandem::fill_exponential(q, reinterpret_cast<double *>(d.p), 0, r).wait();
        tandem::fill_exponential(q, reinterpret_cast<float *>(d.p), 0, r).wait();
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
}

// ---- Conformance cases of the specification ------------------------------------------------

// One case of below.json, fill_below.json, normal.json, exponential.json or choice.json.
struct Case {
    std::string id, kind;
    Key key;
    uint32_t K;
    unsigned w; // draw width
    uint64_t start, n, end, range = 0;
    std::vector<uint64_t> values;
    double ulps = 0, abs = 0; // the tolerance of Float32 normals, 0 for bit equality
    unsigned rejected = 0;
    std::vector<double> weights;
    uint64_t capacity = 0;
    std::vector<uint64_t> cut, alias; // where the case pins the whole table
};

static std::vector<Case> read_cases(const std::string &dir, const char *file) {
    const Json cases = read_json(dir + "/" + file);
    std::vector<Case> out;
    for (const Json &j : cases["cases"].items) {
        Case c;
        c.id = j["id"].text;
        c.kind = j["kind"].text;
        c.key = json_key(j["key"]);
        c.K = (uint32_t)j["K"].u64();
        c.w = c.kind.ends_with("32") ? 32 : 64;
        c.start = j["start"].u64();
        c.n = j["n"].u64();
        if (const Json *r = j.find("range"))
            c.range = r->hex();
        for (const Json &v : j["values"].items)
            c.values.push_back(v.hex());
        if (const Json *t = j.find("tol")) {
            c.ulps = (*t)["ulps"].number();
            c.abs = (*t)["abs"].number();
        }
        if (const Json *r = j.find("rejected"))
            c.rejected = (unsigned)r->u64();
        if (const Json *ws = j.find("weights"))
            for (const Json &x : ws->items)
                c.weights.push_back(std::bit_cast<double>(x.hex()));
        if (const Json *cap = j.find("capacity"))
            c.capacity = cap->hex();
        if (const Json *cut = j.find("cut")) {
            for (const Json &x : cut->items)
                c.cut.push_back(x.hex());
            for (const Json &x : j["alias"].items)
                c.alias.push_back(x.hex());
        }
        // Where the source pins no end: one draw per element, a pair per two Float32 normals.
        const uint64_t p0 = tandem::align_pos(c.start, c.w);
        if (const Json *e = j.find("end"))
            c.end = e->u64();
        else
            c.end = c.kind == "fill_normal_f32" ? p0 + 64 * ((c.n + 1) / 2) : p0 + c.w * c.n;
        out.push_back(c);
    }
    return out;
}

static const Case &case_named(const std::vector<Case> &cases, const std::string &name) {
    for (const Case &c : cases)
        if (c.id.ends_with(" " + name))
            return c;
    throw std::runtime_error("no conformance case " + name);
}

// Bit for bit, or for Float32 normals |y - x| <= ulps 2^-23 |x| + abs.
static bool matches(const Case &c, const std::vector<uint64_t> &got) {
    if (got.size() != c.values.size())
        return false;
    for (size_t i = 0; i < got.size(); i++) {
        if (c.ulps == 0) {
            if (got[i] != c.values[i])
                return false;
            continue;
        }
        double y = sycl::bit_cast<float>((uint32_t)got[i]);
        double x = sycl::bit_cast<float>((uint32_t)c.values[i]);
        if (!(std::abs(y - x) <= c.ulps * 0x1p-23 * std::abs(x) + c.abs))
            return false;
    }
    return true;
}

static std::vector<NormalKernel> case_paths(sycl::queue &q, const Case &c) {
    if (c.kind == "fill_normal_f32")
        return normal_kernels(q);
    return {{Kernel::Chunk, Exchange::Auto, "chunk"}, {Kernel::Tile, Exchange::Auto, "tile"}};
}

// The alias table of a case's weights, built on the host, with its arrays copied to the device.
struct DeviceTable {
    std::vector<uint64_t> cut;
    std::vector<uint32_t> alias;
    tandem::ChoiceTable host{}, dev{};
    bool built;
    Dev<uint64_t> dcut;
    Dev<uint32_t> dalias;
    DeviceTable(sycl::queue &q, const std::vector<double> &weights)
        : cut(weights.size()), alias(weights.size()),
          built(tandem::choice_build(host, weights.data(), weights.size(), cut.data(),
                                     alias.data())),
          dcut(q, cut.size()), dalias(q, alias.size()) {
        if (!cut.empty()) {
            q.memcpy(dcut.p, cut.data(), 8 * cut.size());
            q.memcpy(dalias.p, alias.data(), 4 * alias.size());
            q.wait();
        }
        dev = tandem::ChoiceTable{host.capacity, dcut.p, dalias.p, host.m};
    }
};

// n elements of the case's fill from r on the device by one kernel path, as bit patterns. An
// empty fill goes through the public API onto a sentinel, which it must leave alone.
static std::vector<uint64_t> fill_case(sycl::queue &q, const Case &c, Rng &r, uint64_t n,
                                       const NormalKernel &path) {
    namespace td = tandem::detail;
    Dev<uint64_t> d(q, n + 1);
    auto *u32 = reinterpret_cast<uint32_t *>(d.p);
    auto *f32 = reinterpret_cast<float *>(d.p);
    auto *f64 = reinterpret_cast<double *>(d.p);
    const std::string &k = c.kind;
    const bool choice = k == "fill_choice";
    std::optional<DeviceTable> table;
    if (choice)
        table.emplace(q, c.weights);
    if (n == 0) {
        q.fill(d.p, ~0ull, 1).wait();
        if (choice)
            tandem::fill_choice(q, u32, 0, r, table->dev).wait();
        else if (k == "fill_below_u32")
            tandem::fill_below(q, u32, 0, r, (uint32_t)c.range).wait();
        else if (k == "fill_below_u64")
            tandem::fill_below(q, d.p, 0, r, c.range).wait();
        else if (k == "fill_normal_f64")
            tandem::fill_normal(q, f64, 0, r).wait();
        else if (k == "fill_normal_f32")
            tandem::fill_normal(q, f32, 0, r).wait();
        else if (k == "fill_exponential_f64")
            tandem::fill_exponential(q, f64, 0, r).wait();
        else if (k == "fill_exponential_f32")
            tandem::fill_exponential(q, f32, 0, r).wait();
        CHECK(d.host(1)[0] == ~0ull);
        return {};
    }
    if (choice)
        td::fill_kind<td::choice_idx>(q, td::UsmOut<uint32_t>{u32}, n, r, path.kernel, {}, 0,
                                      table->dev)
            .wait();
    else if (k == "fill_below_u32")
        td::fill_kind<td::below32>(q, td::UsmOut<uint32_t>{u32}, n, r, path.kernel, {}, c.range)
            .wait();
    else if (k == "fill_below_u64")
        td::fill_kind<td::below64>(q, td::UsmOut<uint64_t>{d.p}, n, r, path.kernel, {}, c.range)
            .wait();
    else if (k == "fill_normal_f64")
        td::fill_normal_kind<double>(q, td::UsmOut<double>{f64}, n, r, {}, path.kernel).wait();
    else if (k == "fill_normal_f32")
        td::fill_normal_f32_kind(q, td::UsmOut<float>{f32}, n, r, {}, path.kernel, path.exchange)
            .wait();
    else if (k == "fill_exponential_f64")
        td::fill_kind<td::exp64>(q, td::UsmOut<double>{f64}, n, r, path.kernel, {}).wait();
    else if (k == "fill_exponential_f32")
        td::fill_kind<td::exp32>(q, td::UsmOut<float>{f32}, n, r, path.kernel, {}).wait();
    if (c.w == 64 && !choice)
        return d.host(n);
    std::vector<uint32_t> h(n);
    q.memcpy(h.data(), u32, 4 * n).wait();
    return std::vector<uint64_t>(h.begin(), h.end());
}

// Every case on every kernel path: whole, and cut at elements 1, 7, 20, 21 and n - 1 into
// pieces filled in order on one generator, values and end position. A Float32 normal fill cuts
// only between pairs, at 2, 8, 20 and the largest even element below n, since an odd piece
// drops its last sin half (tandem-spec b31af72). The cases hold the fallbacks
// by global draw index, the empty fills, odd Float32 normal counts and the pair rule.
static void check_fill_cases(sycl::queue &q, const std::vector<Case> &cases) {
    for (const Case &c : cases)
        for (const NormalKernel &path : case_paths(q, c)) {
            Rng r = Rng::from_key(c.key, c.start, c.K);
            bool ok = matches(c, fill_case(q, c, r, c.n, path)) && r.position() == c.end;
            const bool pairs = c.kind == "fill_normal_f32";
            const std::vector<uint64_t> cuts =
                pairs ? std::vector<uint64_t>{2, 8, 20, (c.n - 1) & ~1ull}
                      : std::vector<uint64_t>{1, 7, 20, 21, c.n - 1};
            for (uint64_t cut : cuts) {
                if (c.n == 0 || cut == 0 || cut >= c.n)
                    continue;
                Rng g = Rng::from_key(c.key, c.start, c.K);
                auto a = fill_case(q, c, g, cut, path);
                auto b = fill_case(q, c, g, c.n - cut, path);
                a.insert(a.end(), b.begin(), b.end());
                ok = ok && matches(c, a) && g.position() == c.end;
            }
            CHECK(ok);
            if (!ok)
                std::printf("  %s (%s) differs\n", c.id.c_str(), path.name);
        }
}

// Element i of case a equals element i + shift of case b, both filled here.
static void check_shift(sycl::queue &q, const std::vector<Case> &cases, const char *a,
                        const char *b, size_t shift) {
    const Case &ca = case_named(cases, a), &cb = case_named(cases, b);
    const NormalKernel path = case_paths(q, ca)[0];
    Rng ra = Rng::from_key(ca.key, ca.start, ca.K), rb = Rng::from_key(cb.key, cb.start, cb.K);
    auto va = fill_case(q, ca, ra, ca.n, path), vb = fill_case(q, cb, rb, cb.n, path);
    bool ok = !va.empty();
    for (size_t i = 0; i < va.size() && i + shift < vb.size(); i++)
        ok = ok && va[i] == vb[i + shift];
    CHECK(ok);
}

enum class Scalar { below_u32, below_u64, normal_f64, normal_f32, exp_f64, exp_f32, choice };

static uint64_t scalar_draw(Rng &r, Scalar s, uint64_t range, const tandem::ChoiceTable &t) {
    switch (s) {
    case Scalar::choice:
        return r.choice(t);
    case Scalar::below_u32:
        return r.urand((uint32_t)range);
    case Scalar::below_u64:
        return r.urand64(range);
    case Scalar::normal_f64:
        return sycl::bit_cast<uint64_t>(r.normal());
    case Scalar::normal_f32:
        return sycl::bit_cast<uint32_t>(r.normalf());
    case Scalar::exp_f64:
        return sycl::bit_cast<uint64_t>(r.exponential());
    default:
        return sycl::bit_cast<uint32_t>(r.exponentialf());
    }
}

// The first n values of a case as n scalar draws on the host and in a kernel, and the position
// after them.
static void check_scalars(sycl::queue &q, const Case &c, Scalar s, uint64_t n, uint64_t end) {
    Case head = c;
    head.values.resize(n);
    const DeviceTable table(q, c.weights);
    const tandem::ChoiceTable dev_table = table.dev;
    Rng r = Rng::from_key(c.key, c.start, c.K);
    std::vector<uint64_t> host(n);
    for (uint64_t &v : host)
        v = scalar_draw(r, s, c.range, table.host);
    Dev<uint64_t> d(q, n + 1);
    uint64_t *p = d.p;
    const Key key = c.key;
    const uint64_t start = c.start, range = c.range;
    const uint32_t K = c.K;
    q.single_task([=] {
         Rng g = Rng::from_key(key, start, K);
         for (uint64_t i = 0; i < n; i++)
             p[i] = scalar_draw(g, s, range, dev_table);
         p[n] = g.position();
     }).wait();
    auto dev = d.host(n + 1);
    const uint64_t dev_end = dev.back();
    dev.pop_back();
    bool ok = matches(head, host) && r.position() == end && matches(head, dev) && dev_end == end;
    CHECK(ok);
    if (!ok)
        std::printf("  %s scalar draws differ\n", c.id.c_str());
}

// Range 0 gives 0 and consumes one draw of the width that the interface names.
static void check_range0(sycl::queue &q) {
    Dev<uint64_t> d(q, 1);
    Rng r = Rng::from_key(Rng(42).key(), 33, 32);
    q.fill(d.p, ~0ull, 1).wait();
    tandem::fill_below(q, reinterpret_cast<uint32_t *>(d.p), 1, r, 0u).wait();
    CHECK((uint32_t)d.host(1)[0] == 0 && r.position() == 96);
    tandem::fill_below(q, d.p, 1, r, (uint64_t)0).wait();
    CHECK(d.host(1)[0] == 0 && r.position() == 192);
    CHECK(r.urand(0u) == 0 && r.position() == 224);
    CHECK(r.urand64((uint64_t)0) == 0 && r.position() == 320);
}

static void test_cases(sycl::queue &q, const std::string &dir) {
    const auto below = read_cases(dir, "below.json");
    const auto fill_below = read_cases(dir, "fill_below.json");
    const auto normal = read_cases(dir, "normal.json");
    const auto exponential = read_cases(dir, "exponential.json");
    check_fill_cases(q, fill_below);
    check_fill_cases(q, normal);
    check_fill_cases(q, exponential);
    unsigned rejected = 0;
    for (const Case &c : fill_below)
        rejected += c.rejected;
    CHECK(rejected > 0);

    // Scalar bounded draws retry in sequence. Scalar normals and exponentials equal the fills.
    for (const Case &c : below)
        check_scalars(q, c, c.w == 32 ? Scalar::below_u32 : Scalar::below_u64, c.n, c.end);
    for (const Case &c : normal)
        if (c.kind == "fill_normal_f64" && c.n)
            check_scalars(q, c, Scalar::normal_f64, c.n, c.end);
    for (const Case &c : exponential)
        if (c.n)
            check_scalars(q, c, c.w == 64 ? Scalar::exp_f64 : Scalar::exp_f32, c.n, c.end);
    // A scalar Float32 normal is the cos half of a pair and consumes two draws.
    const Case &nf = case_named(normal, "CROSS_NORMALF");
    check_scalars(q, nf, Scalar::normal_f32, 1, tandem::align_pos(nf.start, 32) + 64);

    // A later start shifts the elements: the fallback follows the global draw index, and a
    // Float32 normal start one pair later shifts the output by one pair.
    check_shift(q, fill_below, "CROSS_BELOW32_AT[4]", "CROSS_BELOW32[4]", 1);
    check_shift(q, fill_below, "CROSS_BELOW64_AT[6]", "CROSS_BELOW64[6]", 1);
    check_shift(q, normal, "CROSS_NORMAL[1]", "CROSS_NORMAL[0]", 1);
    check_shift(q, normal, "CROSS_NORMAL32[2]", "CROSS_NORMAL32[0]", 2);
    check_shift(q, normal, "CROSS_NORMAL32[1]", "CROSS_NORMALF", 0);
    check_range0(q);
}

// ---- Weighted choice ------------------------------------------------------------------------

// Both kernels equal the host's choice() calls at random keys, chunk lengths, positions,
// lengths and output alignments.
static void check_choice_draws(sycl::queue &q, const DeviceTable &t) {
    for (const Trial &tr : trials(57, 12)) {
        Rng r = Rng::from_key(tr.key, tr.pos, tr.K);
        std::vector<uint32_t> want(tr.n);
        for (uint32_t &x : want)
            x = r.choice(t.host);
        for (Kernel kernel : KERNELS) {
            Dev<uint32_t> d(q, tr.n + 4);
            Rng g = Rng::from_key(tr.key, tr.pos, tr.K);
            tandem::detail::fill_kind<tandem::detail::choice_idx>(
                q, tandem::detail::UsmOut<uint32_t>{d.p + tr.shift}, tr.n, g, kernel, {}, 0, t.dev)
                .wait();
            CHECK(first_diff(want, d.host(tr.n, tr.shift)) == SIZE_MAX &&
                  g.position() == r.position());
        }
    }
}

// Every case's table, whole where the case pins it, fill, cuts and scalar draws, the shift of a
// later start, fills at any alignment, buffers, and the weights that build no table.
static void test_choice(sycl::queue &q, const std::string &dir) {
    const auto cases = read_cases(dir, "choice.json");
    size_t tables = 0;
    for (const Case &c : cases) {
        const DeviceTable t(q, c.weights);
        CHECK(t.built && t.host.capacity == c.capacity);
        if (c.cut.empty())
            continue;
        CHECK(t.cut == c.cut &&
              std::equal(t.alias.begin(), t.alias.end(), c.alias.begin(), c.alias.end()));
        tables++;
    }
    CHECK(tables > 0);
    check_fill_cases(q, cases);
    for (const Case &c : cases)
        if (c.n)
            check_scalars(q, c, Scalar::choice, c.n, c.end);
    check_shift(q, cases, "CROSS_CHOICE[1]", "CROSS_CHOICE[0]", 1);

    const DeviceTable t(q, case_named(cases, "choice mixed").weights);
    check_choice_draws(q, t);

    // A buffer of rank 2 fills in linear order with the values and position of a USM fill.
    constexpr size_t a = 37, b = 29;
    Rng u = Rng::from_key(Rng(5).key(), 3, 32), g = u;
    Dev<uint32_t> d(q, a * b);
    tandem::fill_choice(q, d.p, a * b, u, t.dev).wait();
    std::vector<uint32_t> h(a * b);
    {
        sycl::buffer<uint32_t, 2> buf(h.data(), sycl::range<2>(a, b));
        tandem::fill_choice(q, buf, g, t.dev);
    }
    CHECK(first_diff(h, d.host(a * b)) == SIZE_MAX && g.position() == u.position());

    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<std::vector<double>> no_table = {{1, -1}, {1, inf}, {nan, 1}, {0, -0.0}, {}};
    for (const auto &w : no_table)
        CHECK(!DeviceTable(q, w).built);
}

// ---- Dumps of hashes.json and position boundaries ------------------------------------------

static uint64_t fnv1a(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < n; i++)
        h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

// The fills of each dump in order on one generator per start. The Float32 normal hash holds
// for C's polynomials, which the host's normalf2 takes and the fills' native cos and sin do not,
// so that dump runs on the host.
static void test_dumps(sycl::queue &q, const Json &hashes) {
    for (const Json &dump : hashes["dumps"].items) {
        const Key key = json_key(dump["key"]);
        uint64_t h = 0xcbf29ce484222325ull, end = 0;
        for (const Json &start : dump["starts"].items) {
            Rng r = Rng::from_key(key, start.u64(), (uint32_t)dump["K"].u64());
            for (const Json &draw : dump["draws"].items) {
                const std::string &kind = draw["kind"].text;
                const size_t n = draw["n"].u64();
                if (kind == "fill_normal_f32") {
                    std::vector<float> z = sequential_normals(r, n);
                    h = fnv1a(h, z.data(), 4 * n);
                    continue;
                }
                Dev<double> d(q, n);
                size_t size = 8;
                if (kind == "fill_normal_f64")
                    tandem::fill_normal(q, d.p, n, r).wait();
                else if (kind == "fill_exponential_f64")
                    tandem::fill_exponential(q, d.p, n, r).wait();
                else if (kind == "fill_exponential_f32")
                    tandem::fill_exponential(q, reinterpret_cast<float *>(d.p), n, r).wait(),
                        size = 4;
                std::vector<uint8_t> bytes(n * size);
                q.memcpy(bytes.data(), d.p, bytes.size()).wait();
                h = fnv1a(h, bytes.data(), bytes.size());
            }
            end = r.position();
        }
        const Json *want_end = dump.find("end");
        bool ok = h == dump["fnv1a"].hex() && (!want_end || end == want_end->u64());
        CHECK(ok);
        if (!ok)
            std::printf("  %s: hash %016llx\n", dump["id"].text.c_str(), (unsigned long long)h);
    }
}

// A complex value whose real part ends a block takes its imaginary part from the next block.
static void check_complex_straddle(sycl::queue &q) {
    const Key key = Rng(42).key();
    const Rng at0 = Rng::from_key(key, 0, 32);
    Dev<std::complex<double>> zd(q, 1);
    Dev<std::complex<float>> zf(q, 1);
    Rng r = Rng::from_key(key, 64, 32);
    tandem::fill(q, zd.p, 1, r).wait();
    CHECK(r.position() == 192);
    r.set_position(96);
    tandem::fill(q, zf.p, 1, r).wait();
    CHECK(r.position() == 160);
    auto d = zd.host(1);
    auto f = zf.host(1);
    CHECK(d[0].real() == at0.at_drand(1) && d[0].imag() == at0.at_drand(2));
    CHECK(f[0].real() == at0.at_frand(3) && f[0].imag() == at0.at_frand(4));
}

// Random access on the host and in a kernel equals the fill from the same position, from starts
// on both sides of block, row and chunk boundaries. A chunk at K = 8 is 8192 bits.
static void check_random_access(sycl::queue &q) {
    const Key key = Rng(42).key();
    constexpr size_t n = 64;
    for (uint64_t p : {0ull, 100ull, 127ull, 128ull, 1000ull, 1023ull, 1024ull, 6000ull, 8191ull,
                       8192ull, 8193ull, 16389ull}) {
        auto u64 = device_fill<uint64_t>(q, key, p, 8, n, Kernel::Chunk);
        auto u32 = device_fill<uint32_t>(q, key, p, 8, n, Kernel::Chunk);
        Dev<uint64_t> d(q, 2 * n);
        uint64_t *v = d.p;
        q.single_task([=] {
             const Rng r = Rng::from_key(key, p, 8);
             for (size_t i = 0; i < n; i++) {
                 v[i] = r.at_urand64(i);
                 v[n + i] = r.at_urand(i);
             }
         }).wait();
        auto dev = d.host(2 * n);
        const Rng r = Rng::from_key(key, p, 8);
        bool ok = true;
        for (size_t i = 0; i < n; i++)
            ok = ok && r.at_urand64(i) == u64[i] && dev[i] == u64[i] && r.at_urand(i) == u32[i] &&
                 dev[n + i] == u32[i];
        CHECK(ok);
    }
}

// A generator accepts start 2^63 - 1 and rejects starts 2^63 and 2^64 - 1 without changing
// state. A UInt64 draw at 2^63 - 1 aligns to 2^63, on the host and in a fill. A fill whose end
// reaches 2^64 throws before it writes or moves the position, and one that ends just below runs.
static void check_position_bounds(sycl::queue &q) {
    const Key key = Rng(42).key();
    const uint64_t top = 1ull << 63;
    Rng r = Rng::from_key(key, top - 1, 32);
    const Rng start = r;
    CHECK(r.position() == top - 1);
    CHECK(!r.set_position(top) && r == start);
    CHECK(!r.set_position(~0ull) && r == start);
    const uint64_t x = r.urand64();
    CHECK(r.position() == top + 64 && x == start.at_urand64(0));
    uint64_t end;
    auto v = device_fill<uint64_t>(q, key, top - 1, 32, 1, Kernel::Auto, 0, &end);
    CHECK(v[0] == x && end == top + 64);

    Dev<uint64_t> d(q, 4);
    q.fill(d.p, ~0ull, 4).wait();
    Rng g = Rng::from_key(key, 0, 32);
    g.advance_to(~0ull - 255);
    bool threw = false;
    try {
        tandem::fill(q, d.p, 4, g);
    } catch (const std::overflow_error &) {
        threw = true;
    }
    auto h = d.host(4);
    CHECK(threw && g.position() == ~0ull - 255 &&
          std::all_of(h.begin(), h.end(), [](uint64_t y) { return y == ~0ull; }));
    tandem::fill(q, d.p, 3, g).wait();
    CHECK(g.position() == ~0ull - 63);
}

int main(int argc, char **argv) {
    const std::string dir = argc > 1 ? argv[argc - 1] : "tests/conformance";
    const Json hashes = read_json(dir + "/hashes.json");
    sycl::queue q;
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    test_vectors(q);
    test_streams(q, hashes);
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
    test_normal64(q);
    test_normal32(q);
    test_exponential(q);
    test_positions(q);
    test_cases(q, dir);
    test_choice(q, dir);
    test_dumps(q, hashes);
    check_complex_straddle(q);
    check_random_access(q);
    check_position_bounds(q);
    if (failures) {
        std::printf("%ld of %ld checks failed\n", failures, checks);
        return 1;
    }
    std::printf("ok: %ld checks\n", checks);
    return 0;
}
