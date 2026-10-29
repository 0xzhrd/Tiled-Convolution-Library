#pragma once

#ifndef __CUDACC__
#error "conv_halo.cuh must be compiled by nvcc"
#endif

#include "boundary.hpp"
#include "filter_bank.cuh"
#include "config.hpp"
#include "cudacheck.hpp"
#include "reference.hpp"

#include <stdexcept>
#include <cstddef>
#include <cuda_runtime.h>

namespace tiled_convolution
{
    template<int R, int TX, int TY>
    struct HaloGeometry //every size the kernel needs compute donce
    {
        static constexpr int filter_w = 2 * R + 1;
        static constexpr int in_w = TX + 2 * R;
        static constexpr int in_h = TY + 2 * R;

        static constexpr int threads = TX * TY;
        static constexpr int stride = (TX % 32 == 0) ? in_w + (((TX - in_w) % 32) + 32) % 32;
        static constexpr std::size_t mem_bytes = sizeof(float) * static_cast<size_t>(in_h) * static_cast<size_t>(stride);

        static_assert(R >= 0 && R <= max_filter_radius, "the radius in the bank should be greater than 0 or less than max_filter width" + std::to_string(max_filter_width));
        

    };
}