// cuRAND Philox4x32-10 for the bench's comparison rows, through the CUDA runtime that
// AdaptiveCpp's CUDA backend shares, so a USM device pointer is a cuRAND output. Plain C++: this
// file holds no SYCL code. Each call returns after the device finishes.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>
#include <curand.h>

#include "bench_curand.hpp"

namespace {
curandGenerator_t generator() {
    static curandGenerator_t g = [] {
        curandGenerator_t h;
        curandCreateGenerator(&h, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        curandSetPseudoRandomGeneratorSeed(h, 42);
        return h;
    }();
    return g;
}
} // namespace

int curand_version() {
    int v = 0;
    curandGetVersion(&v);
    return v;
}

void curand_fill(CurandCall call, void *out, size_t n, uint64_t offset) {
    curandGenerator_t g = generator();
    curandSetGeneratorOffset(g, offset);
    switch (call) {
    case CurandCall::Generate: curandGenerate(g, static_cast<unsigned *>(out), n); break;
    case CurandCall::Uniform: curandGenerateUniform(g, static_cast<float *>(out), n); break;
    case CurandCall::UniformDouble:
        curandGenerateUniformDouble(g, static_cast<double *>(out), n);
        break;
    case CurandCall::Normal: curandGenerateNormal(g, static_cast<float *>(out), n, 0.0f, 1.0f); break;
    case CurandCall::NormalDouble:
        curandGenerateNormalDouble(g, static_cast<double *>(out), n, 0.0, 1.0);
        break;
    }
    cudaDeviceSynchronize();
}
