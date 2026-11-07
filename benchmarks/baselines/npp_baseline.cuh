/*

*/

#pragma once

#include "t_convl/boundary.hpp"
#include "t_convl/device_buffer.hpp"
#include "t_convl/filter_bank.cuh"
#include "t_convl/reference.hpp"
#include "common_baseline.hpp"
#include <string>
#include <vector>
#include <sstream>
#include <cstddef>
#include <stdexcept>
#include <npp.h>
#include <cuda_runtime.h>

#define T_CONVL_CHECK_NPP(call)                                                         \
    do                                                                                  \
    {                                                                                   \
        const NppStatus  status__ = (call);                                             \
        if(status__ < 0)                                                                \
        {                                                                               \
            std::ostringstream oss__;                                                   \
            oss__ << "NPP error at: " << __FILE__ << ':' << __LINE__ << " in '  "       \
            << #call << "': status " << static_cast<int>(status__);                     \
            throw std::runtime_error(oss__.str());                                      \
        }                                                                               \
    } while(0)

namespace tiled_convolution
{
    namespace baseline
    {
        class NppConv
        {
            public:
                NppConv(const Filter &filter, Image<const float> in, Image<float> out, int batch, std::ptrdiff_t in_image_stride, std::ptrdiff_t out_image_stride, 
                    BorderMode border, cudaStream_t stream, bool flip_kernel = true) : in_(in), out_(out), batch_(batch), in_stride_(in_image_stride),
                                                                                        out_stride_(out_image_stride), radius_(filter.radius()), width_(filter.width()) 
                {
                    if(border != BorderMode::Clamp)
                        throw Unsupported("NPP FilterBorder allows replicate only (border mode " + std::string(border_name(border)) + ")");

                    const std::vector<float> coeffs = flip_kernel ? reference::flip(filter.data(), radius_) : filter.taps();
                    
                    kernel_ = DeviceBuffer<float>(coeffs.size());
                    kernel_.upload(coeffs, stream);

                    ctx_ = NppStreamContext();
                    ctx_.hStream = stream;
                    T_CONVL_CHECK_CUDA(cudaGetDevice(&ctx_.nCudaDeviceId));
                    cudaDeviceProp props;
                    T_CONVL_CHECK_CUDA(cudaGetDeviceProperties(&props, ctx_.nCudaDeviceId));
                    ctx_.nMultiProcessorCount = props.multiProcessorCount;
                    ctx_.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
                    ctx_.nMaxThreadsPerBlock = props.maxThreadsPerBlock;
                    ctx_.nSharedMemPerBlock = props.sharedMemPerBlock;
                    ctx_.nCudaDevAttrComputeCapabilityMajor = props.major;
                    ctx_.nCudaDevAttrComputeCapabilityMinor = props.minor;
                    T_CONVL_CHECK_CUDA(cudaStreamGetFlags(stream, &ctx_.nStreamFlags));
                }

                NppConv(const NppConv &) = delete;
                NppConv &operator = (const NppConv &) = delete;

                void run()
                {
                    const NppiSize size{in_.width, in_.height};
                    const NppiSize ksize{width_, width_};
                    const NppiPoint origin{0, 0};
                    const NppiPoint anchor{radius_, radius_};
                    const Npp32s src_step = static_cast<Npp32s>(static_cast<std::size_t>(in_.pitch) * sizeof(float));
                    const Npp32s dst_step = static_cast<Npp32s>(static_cast<std::size_t>(out_.pitch) * sizeof(float));

                    for(int x = 0; x < batch_; ++x)
                    {
                        const float *src = in_.data + static_cast<std::ptrdiff_t>(x) * in_stride_;
                        float *dst = out_.data + static_cast<std::ptrdiff_t>(x) * out_stride_;
                        T_CONVL_CHECK_NPP(nppiFilterBorder_32f_C1R_Ctx(src, src_step, size, origin, dst, dst_step, size, kernel_.data(), ksize, anchor, NPP_BORDER_REPLICATE, ctx_));
                    }
                }
                static const char *name()
                {
                    return("Npp");
                }

            private:
                Image<const float> in_;
                Image<float> out_;
                int batch_;
                std::ptrdiff_t in_stride_;
                std::ptrdiff_t out_stride_;
                int radius_;
                int width_;
                DeviceBuffer<float> kernel_;
                NppStreamContext ctx_;
        };
    }
}