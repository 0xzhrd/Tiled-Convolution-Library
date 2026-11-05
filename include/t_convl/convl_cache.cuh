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
        static constexpr int filter_w = 2 * R + 1;
        static constexpr int threads = TX * TY;
        std::size_t smem_bytes = sizeof(float) * TY * TY;
        
        static_assert(R >= 0 && R <= max_filter_radius, "CacheGeometry: radius is not within the established limits [1, 8]");
        static_assert(threads <= 1024, "CacheGeometry: there should be a maximum of 1024 threads per block");
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

    template <int R, typename Border, int TX = 16, int TY = 16, typename T>
    void convl_cache(Image<const T> in, Image<float> out, int slot, cudaStream_t stream, int batch = 1, std::ptrdiff_t in_image_stride = 0, std::ptrdiff_t out_image_stride = 0)
    {
        if(in.width != out.width || in.height != out.height)
            throw std::invalid_argument("convl_cache: input " + std::to_string(in.width) + "x" + std::to_string(in.height) + " != output " + std::to_string(out.width) + "x" + std::to_string(out.height));

        if(slot < 0 || slot >= filter_slots)
            throw std::out_of_range("convl_cache: slot " + std::to_string(slot) + " outside [0, " + std::to_string(filter_slots) + "]");

        if(batch > 1)
        {
            const std::ptrdiff_t in_min = static_cast<std::ptrdiff_t>(in.pitch) * in.height;
            const std::ptrdiff_t out_min = static_cast<std::ptrdiff_t>(out.pitch) * out.height;

            if(in_image_stride< in_min || out_image_stride < out_min)
                throw std::invalid_argument("convl_cache: batched images overlap(stride < pitch * height)");
        }
        check_border_contract<Border>(in.width, in.height, R);

        const dim3 block(TX, TY, 1);
        const dim3 grid((in.width + TX - 1) / TX, (in.height + TY - 1) / TY, batch);

        conv_cache_kernel<R, Border, TX, TY, T><<<grid, block, 0, stream>>>(in, out, in_image_stride, out_image_stride, slot);
        T_CONVL_CHECK_CUDA(cudaGetLastError());
    }
}
