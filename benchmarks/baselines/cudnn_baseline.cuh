/*
cudnnConvolutionForward on a 1-channel NCHW batch
*/

#pragma once

#include "t_convl/boundary.hpp"
#include "t_convl/device_buffer.hpp"
#include "t_convl/filter_bank.cuh"
#include "common_baseline.hpp"
#include <cudnn.h>
#include <cstddef>
#include <sstream>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

#define T_CONVL_CHECK_CUDNN(call)                                                       \
    do                                                                                  \
    {                                                                                   \
        const cudnnStatus_t status__ = (call);                                          \
        if(status__ != CUDNN_STATUS_SUCCESS)                                            \
        {                                                                               \
            std::ostringstream oss__;                                                   \
            oss__ << "CUDNN error at: " << __FILE__ << ':' << __LINE__ << " in '"       \
            << #call << "': " << cudnnGetErrorString(status__);                         \
            throw std::runtime_error(oss__.str());                                      \
        }                                                                               \
    } while(0)

namespace tiled_convolution
{
    namespace baseline
    {
        class CudnnConv
        {
            public: 
                CudnnConv(const Filter &filter, Image<const float> in, Image<float> out, int batch, std::ptrdiff_t in_image_stride, std::ptrdiff_t out_image_stride, BorderMode border, cudaStream_t stream) : in_(in), out_(out)
                {
                    if(border != BorderMode::Zero)
                    {
                        throw Unsupported("cuDNN pads with zeroes only (border mode " + std::string(border_name(border)) + ")");
                    }
                    const int r = filter.radius();
                    const int filter_w = filter.width();

                    T_CONVL_CHECK_CUDNN(cudnnCreate(&handle_));
                    T_CONVL_CHECK_CUDNN(cudnnSetStream(handle_, stream));

                    const int n_in = static_cast<int>(batch > 1 ? in_image_stride : static_cast<std::ptrdiff_t>(in.pitch) * in.height);
                    const int n_out = static_cast<int>(batch > 1 ? out_image_stride : static_cast<std::ptrdiff_t>(out.pitch) * out.height);

                    T_CONVL_CHECK_CUDNN(cudnnCreateTensorDescriptor(&in_desc_));
                    T_CONVL_CHECK_CUDNN(cudnnSetTensor4dDescriptorEx(in_desc_, CUDNN_DATA_FLOAT, batch, 1, in.height, in.width, n_in, n_in, in.pitch, 1));
                    T_CONVL_CHECK_CUDNN(cudnnCreateTensorDescriptor(&out_desc_));
                    T_CONVL_CHECK_CUDNN(cudnnSetTensor4dDescriptorEx(out_desc_, CUDNN_DATA_FLOAT, batch, 1, out.height, out.width, n_out, n_out, out.pitch, 1));

                    T_CONVL_CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc_));
                    T_CONVL_CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, filter_w, filter_w));

                    T_CONVL_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));
                    T_CONVL_CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc_, r, r, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
                    T_CONVL_CHECK_CUDNN(cudnnSetConvolutionMathType(conv_desc_, CUDNN_FMA_MATH));

                    weights_ = DeviceBuffer<float>(filter.size());
                    T_CONVL_CHECK_CUDA(cudaMemcpyAsync(weights_.data(), filter.data(), weights_.bytes(), cudaMemcpyHostToDevice, stream));

                    cudnnConvolutionFwdAlgoPerf_t perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
                    int returned = 0;
                    T_CONVL_CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(handle_, in_desc_, filter_desc_, conv_desc_, out_desc_, CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned, perf));

                    bool found = false;
                    for(int i = 0; i < returned && !found; ++i)
                    {
                        if(perf[i].status == CUDNN_STATUS_SUCCESS)
                        {
                            algo_ = perf[i].algo;
                            workspace_size_ = perf[i].memory;
                            found = true;
                        }
                    }
                    if(!found)
                    {
                        throw Unsupported("cuDNN did not find a valid algorithm for this configuration");
                    }

                    if(workspace_size_ > 0)
                    {
                        workspace_ = DeviceBuffer<unsigned char>(workspace_size_);
                    }
                }

                ~CudnnConv()
                {
                    if(conv_desc_ != nullptr) cudnnDestroyConvolutionDescriptor(conv_desc_);
                    if(filter_desc_ != nullptr) cudnnDestroyFilterDescriptor(filter_desc_);
                    if(out_desc_ != nullptr) cudnnDestroyTensorDescriptor(out_desc_);
                    if(in_desc_ != nullptr) cudnnDestroyTensorDescriptor(in_desc_);
                    if(handle_ != nullptr) cudnnDestroy(handle_);
                }

                CudnnConv(const CudnnConv &) = delete;
                CudnnConv &operator = (const CudnnConv &) = delete;

                void run()
                {
                    const float alpha = 1.0f;
                    const float beta = 0.0f;
                    T_CONVL_CHECK_CUDNN(cudnnConvolutionForward(handle_, &alpha, in_desc_, in_.data, filter_desc_, weights_.data(), conv_desc_, algo_, workspace_.data(), workspace_size_, &beta, out_desc_, out_.data));  
                }

            static const char *name() 
            {
                return("cuDNN");
            }
            
            int algorithm() const 
            {
                return static_cast<int>(algo_);
            }
            
            private:
                Image<const float> in_;
                Image<float> out_;

                cudnnHandle_t handle_ = nullptr;
                cudnnTensorDescriptor_t in_desc_ = nullptr;
                cudnnTensorDescriptor_t out_desc_ = nullptr;
                cudnnFilterDescriptor_t filter_desc_ = nullptr;
                cudnnConvolutionDescriptor_t conv_desc_ = nullptr;

                cudnnConvolutionFwdAlgo_t algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
                std::size_t workspace_size_ = 0;
                DeviceBuffer<float> weights_;
                DeviceBuffer<unsigned char> workspace_;
        };
    }
}