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

        //prob something missing here
        
        const bool reachable = ox <= src.width - 1 + R && oy <= src.height - 1 + R;

        tile[ty * TX + tx] = reachable ? static_cast<float>(Border::load(src, ox, oy)) : 0.0f;
        __syncthreads();

        if(ox >= dst.width || oy >= dst.height)
        {
            return;
        }

        float acc = 0.0f;
        #pragma unroll 1
        for(int dy = 0; dy < G::filter_w; ++dy)
        {
            const int ly = ty + dy - R;
            const int gy = oy + dy - R;
            const bool row_in = static_cast<unsigned>(ly) < static_cast<unsigned>(TY);
            #pragma unroll
            for(int dx = 0; dx < G::filter_w; ++dx)
            {
                const int lx = tx + dx - R;
                const float p = (row_in && static_cast<unsigned>(lx) < static_cast<unsigned>(TX)) ? tile[ly * TX + lx] : static_cast<float>(Border::load(src, ox + dx - R, gy));
                acc += filter_tap(slot, dy * G::filter_w + dx) * p;

            }
        }
        dst.row(oy)[ox] = acc;
    }

    
}