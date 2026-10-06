// The cuRAND calls of the bench, see bench_curand.cpp.
#pragma once

#include <cstddef>
#include <cstdint>

enum class CurandCall { Generate, Uniform, UniformDouble, Normal, NormalDouble };

int curand_version();
/* n values of the call into device memory, from the generator's absolute offset. */
void curand_fill(CurandCall call, void *out, size_t n, uint64_t offset = 0);
