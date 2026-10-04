// Fill throughput on the default SYCL device, GiB/s written. Each row runs eight back-to-back
// fills on an in-order queue and one wait, after a half-second warm-up, and keeps the minimum
// time of 15 runs. Every fill starts at position 0 unless its label says otherwise. An argument
// keeps the rows whose label contains it.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>

#include <tandem/sycl.hpp>

using clock_type = std::chrono::steady_clock;
using tandem::detail::Kernel;

static double seconds_since(clock_type::time_point t0) {
    return std::chrono::duration<double>(clock_type::now() - t0).count();
}

static double best_gibs(sycl::queue &q, size_t bytes, const std::function<void()> &fill) {
    constexpr int per_run = 8, runs = 15;
    double best = 1e30;
    for (int r = 0; r < runs; r++) {
        auto t0 = clock_type::now();
        for (int i = 0; i < per_run; i++)
            fill();
        q.wait();
        best = std::min(best, seconds_since(t0));
    }
    return (double)bytes * per_run / best / (1024.0 * 1024.0 * 1024.0);
}

int main(int argc, char **argv) {
    const char *only = argc > 1 ? argv[1] : "";
    sycl::queue q{sycl::property::queue::in_order()};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const size_t max_n = (size_t)1 << 28;
    void *buf = sycl::malloc_device(max_n * 16, q);
    const tandem::Rng rng(42);

    auto fill = [&](auto tag, size_t n, Kernel kernel) {
        using E = decltype(tag);
        using P = tandem::detail::plan<E>;
        using O = tandem::detail::out_of<E>;
        tandem::Rng r = rng;
        tandem::detail::fill_kind<typename P::kind>(
            q, tandem::detail::UsmOut<O>{static_cast<O *>(buf)}, (uint64_t)n * P::parts, r,
            kernel, {});
    };

    for (auto t0 = clock_type::now(); seconds_since(t0) < 0.5;) {
        fill(uint32_t{}, max_n, Kernel::Auto);
        q.wait();
    }

    std::printf("%-40s %10s %10s\n", "", "2^26", "2^28");
    auto row = [&](const char *label, size_t elem_bytes, auto body) {
        if (!std::strstr(label, only))
            return;
        std::printf("%-40s", label);
        for (size_t n : {(size_t)1 << 26, max_n})
            std::printf(" %10.0f", best_gibs(q, n * elem_bytes, [&] { body(n); }));
        std::printf("\n");
        std::fflush(stdout);
    };
    row("fill uint32_t", 4, [&](size_t n) { fill(uint32_t{}, n, Kernel::Auto); });
    row("fill uint64_t", 8, [&](size_t n) { fill(uint64_t{}, n, Kernel::Auto); });
    row("fill float", 4, [&](size_t n) { fill(float{}, n, Kernel::Auto); });
    row("fill double", 8, [&](size_t n) { fill(double{}, n, Kernel::Auto); });
    row("fill uint32_t, chunk kernel", 4, [&](size_t n) { fill(uint32_t{}, n, Kernel::Chunk); });
    row("fill uint64_t, chunk kernel", 8, [&](size_t n) { fill(uint64_t{}, n, Kernel::Chunk); });
    row("fill float, chunk kernel", 4, [&](size_t n) { fill(float{}, n, Kernel::Chunk); });
    row("fill double, chunk kernel", 8, [&](size_t n) { fill(double{}, n, Kernel::Chunk); });
    row("fill uint16_t", 2, [&](size_t n) { fill(uint16_t{}, n, Kernel::Auto); });
    row("fill sycl::half", 2, [&](size_t n) { fill(sycl::half{}, n, Kernel::Auto); });
    row("fill uint8_t", 1, [&](size_t n) { fill(uint8_t{}, n, Kernel::Auto); });
    row("fill bool", 1, [&](size_t n) { fill(bool{}, n, Kernel::Auto); });
    row("fill std::complex<double>", 16,
        [&](size_t n) { fill(std::complex<double>{}, n, Kernel::Auto); });
    row("fill_below uint32_t, range 1000", 4, [&](size_t n) {
        tandem::Rng r = rng;
        tandem::fill_below(q, static_cast<uint32_t *>(buf), n, r, 1000u);
    });
    row("fill_below uint64_t, range 1000", 8, [&](size_t n) {
        tandem::Rng r = rng;
        tandem::fill_below(q, static_cast<uint64_t *>(buf), n, r, 1000u);
    });
    row("fill_below uint32_t, range 1000, chunk kernel", 4, [&](size_t n) {
        tandem::Rng r = rng;
        tandem::detail::fill_kind<tandem::detail::below32>(
            q, tandem::detail::UsmOut<uint32_t>{static_cast<uint32_t *>(buf)}, n, r,
            Kernel::Chunk, {}, 1000u);
    });
    /* Normal fills from stream word w: an odd w for float or an odd Float64 draw for double
     * shifts the pairs within the blocks, and w = 4 or 6 puts the first block at lane 1. */
    auto normal_row = [&](const char *label, auto tag, uint64_t w) {
        using E = decltype(tag);
        row(label, sizeof(E), [&](size_t n) {
            tandem::Rng r = tandem::Rng::from_key(rng.key(), 32 * w, 32);
            tandem::fill_normal(q, static_cast<E *>(buf), n, r);
        });
    };
    normal_row("fill_normal float", float{}, 0);
    normal_row("fill_normal float, odd start", float{}, 1);
    normal_row("fill_normal float, start at word 4", float{}, 4);
    normal_row("fill_normal double", double{}, 0);
    normal_row("fill_normal double, odd start", double{}, 2);
    normal_row("fill_normal double, start at word 6", double{}, 6);
    sycl::free(buf, q);
}
