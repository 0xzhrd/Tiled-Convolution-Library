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
#include <string>
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
        static constexpr int stride = (TX % 32 == 0) ? in_w : in_w + (((TX - in_w) % 32) + 32) % 32;
        static constexpr std::size_t mem_bytes = sizeof(float) * static_cast<size_t>(in_h) * static_cast<size_t>(stride);

        static_assert(R >= 0 && R <= max_filter_radius, "adius in the bank is outside [0, max_filter_radius]" + std::to_string(max_filter_width));
        static_assert(threads % 32 == 0, "block size is not a multiple of 32");
        static_assert(threads <= 1024, "thread block size surpasses hardware limits(1024 threads per block)");
        static_assert(TX % 32 == 0 || 32 % TX == 0, "TX must be divide 32 or be a multiple of 32");
    };

    template <int R, typename Border, int TX, int TY, typename T>
    __global__ void __launch_bounds__(TX * TY) conv_halo_kernel(Image<const T> in, Image<float> out, std::ptrdiff_t in_image_stride, std::ptrdiff_t out_image_stride, int slot)
    {
        using G = HaloGeometry<R, TX, TY>; //local aliad so its G::in_w rather than 'full template name'::in_w
        __shared__ float tile[G::in_h * G::stride]; 

        Image<const T> src = in;
        Image<float> dst = out;
        src.data += static_cast<std::ptrdiff_t>(blockIdx.z) * in_image_stride;
        dst.data += static_cast<std::ptrdiff_t>(blockIdx.z) * out_image_stride;

        const int tx = static_cast<int>(threadIdx.x);
        const int ty = static_cast<int>(threadIdx.y);

        const int t_id = ty * TX + tx;
        const int x0 = static_cast<int>(blockIdx.x) * TX - R;
        const int y0 = static_cast<int>(blockIdx.y) * TY - R;

        for(int i = t_id; i < G::in_w * G::in_h; i += G::threads)
        {
            const int ly = i / G::in_w;
            const int lx = i - ly * G::in_w;

            tile[ly * G::stride + lx] = static_cast<float>(Border::load(src, x0 + lx, y0 + ly));
        }
        __syncthreads();

        const int ox = static_cast<int>(blockIdx.x) * TX + tx;
        const int oy = static_cast<int>(blockIdx.y) * TY + ty;

        if(ox >= dst.width || oy >= dst.height)     return;

        float acc = 0.0f; 
        #pragma unroll 1
        for(int dy = 0; dy < G::filter_w; ++dy)
        {
            const float *tile_row = tile + (ty + dy) * G::stride + tx;
            #pragma unroll
            for(int dx = 0; dx < G::filter_w; ++dx)
            {
                acc += filter_tap(slot, dy * G::filter_w + dx) * tile_row[dx];
            }
        }
        dst.row(oy)[ox] = acc;
    }

    template <int R, typename Border, int TX = 16, int TY = 16, typename T>
    void conv_halo(Image<const T> in, Image<float> out, int slot, cudaStream_t stream, int batch = 1, std::ptrdiff_t in_image_stride = 0, std::ptrdiff_t out_image_stride = 0)
    {
        if(in.width != out.width || in.height != out.height)
            throw std::invalid_argument("input width " + std::to_string(in.width) + " is not the same as output width " + std::to_string(out.width)
                                        + "or the input height" + std::to_string(in.height) + "is not the same as the output height " + std::to_string(out.height));
                                    
        if(slot < 0 || slot >= filter_slots)
            throw std::out_of_range("slot is not within range" + std::to_string(filter_slots));

        if(batch < 1 || batch > 65535)
            throw std::invalid_argument("batch size is out of range");
        if(batch > 1)
        {
            const std::ptrdiff_t in.min = static_cast<ptrdiff_t>(in.pitch) * in.height; 
            const std::ptrdiff_t out.min = static_cast<ptrdiff_t>(out.pitch) * out.height;
            if(in_image_stride < in.min)
                throw std::out_of_range("input image stride is less than input minimum");
            
            if(out_image_stride < out.min)
                throw std::out_of_range("output image stride is less than the output minimum");
        }

        check_border_contract<Border>(in.width, in.height, R);

        const dim3 blockDim(TX, TY, 1);
        const dim3 gridDim((in.width + TX - 1) / TX, (in.height + TY - 1) / TY, batch);

        conv_halo_kernel<R, Border, TX, TY, T><<<gridDim, blockDim, 0, stream>>>(in, out, in_image_stride, out_image_stride, slot);

        T_CONVL_CHECK_CUDA(cudaGetLastError());        
    }
}