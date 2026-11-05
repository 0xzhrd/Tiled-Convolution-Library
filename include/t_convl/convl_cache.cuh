/*

*/
#pragma once

#ifndef __CUDACC__
#error "convl_cache.cuh must be compiled by nvcc"
#endif

#include "t_convl/cudacheck.hpp"
#include "t_convl/filter_bank.cuh"
#include "t_convl/boundary.hpp"
#include "t_convl/config.hpp"
#include <cstddef>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace tiled_convolution
{
    template <int R, int TX, int TY>
    struct CacheGeometry
    {
        static constexpr int fw = 2 * R + 1;
        static constexpr int threads = TX * TY;
        std::size_t smem_bytes = sizeof(float) * TY * TY;
        
        static_assert(R >= 1 && R <= 8, "CacheGeometry: radius is not within the established limits [1, 8]");
        static_assert(threadsa <= 1024, "CacheGeometry: there should be a maximum of 1024 threads per block");
        static_assert(TX % 32 == 0 || 32 % TX == 0, "CacheGeometry: TX must be divisible or must be a multiple of 32");
        static_assert(threads % 32 == 0, "CacheGeometry: block size must be a multiple of a warp");
    };  
    template <int R, typename Border, int TX, int TY, typename T>
    __global__ void __launch_bounds__(TX * TY) conv_cache_kernel(Image<const T> in, Image<float> out, std::ptrdiff_t in_image_stride, std::ptrdiff_t out_image_stride, int slot)
    {//launcher can treat this and conv_halo_kernel as interchangeable
        using G = CacheGeometry<R, TX, TY>;
        __shared__ float tile[TX * TY];

        Image<const T> src = in;
        Image<float> dst = out;
        src.data += static_cast<std::ptrdiff_t>(blockIdx.z) * in_image_stride;
        dst.data += static_cast<std::ptrdiff_t>(blockIdx.z) * out_image_stride;

        const int tx = static_cast<int>(threadIdx.x);
        const int ty = static_cast<int>(threadIdx.y);

        const int ox = static_cast<int>(blockIdx.x) * TX + tx;
        const int oy = static_cast<int>(blockIdx.y) * TY + ty;
        if(ox >= dst.width || oy >= dst.height)
        {
            return;
        }


        //prob something missing here
        
        
    }
}