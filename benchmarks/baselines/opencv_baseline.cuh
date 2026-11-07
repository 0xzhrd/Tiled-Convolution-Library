/*

*/

#pragma once

#ifdef T_CONVL_WITH_OPENCV

#include "t_convl/boundary.hpp"
#include "t_convl/filter_bank.cuh"
#include "common_baseline.hpp"

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/filters.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core.hpp>

#include <cuda_runtime.h>
#include <cstddef>
#include <string>
#include <vector>

namespace tiled_convolution
{
    namespace baseline
    {
        class OpenCVConv
        {
            public: 
                OpenCVConv(const Filter &filter, Image<const float> in, Image<float> out, int batch, std::ptrdiff_t in_image_stride, std::ptrdiff_t out_image_stride, 
                    BorderMode border, cudaStream_t stream)
                {
                    const cv::Mat kernel = cv::Mat(filter.width(), filter.width(), CV_32F, const_cast<float *>(filter.data())).clone();
                    try
                    {
                        filter_ = cv::cuda::createLineearFilter(CV_32FC1, CV32FC1, kernel,cv::Point(-1, -1), to_opencv(border), cv::Scalar::all(0));
                    }
                    catch(const cv::Exception &e)
                    {
                        throw Unsupported("OpenCV createLinearFilter rejected this configuration: " + e.what());
                    }

                    stream_ = cv::cuda::StreamAccessor::wrapStream(stream);

                    const std::size_t in_step = static_cast<std::size_t>(in.pitch) * sizeof(float);
                    const std::size_t out_step = static_cast<std::size_t>(out.pitch) * sizeof(float);
                    for(int x = 0; x < batch; ++b)
                    {
                        float *src = const_cast<float *>(in.data) + static_cast<std::ptrdiff_t>(x) * in_image_stride;
                        float *dst = out.data + static_cast<std::ptrdiff_t>(x) * out_image_stride;
                        src_.emplace_back(in.height, in.width, CV_32FC1, src, in_step);
                        dst.emplace_back(out.height, out.width, CV_32FC1, dst, out_step);
                    }
                }
                OpenCVConv(const OpenCVConv &) = delete;
                OpenCVConv &operator = (const OpenCVConv &) = delete;

                void run()
                {
                    for(std::size_t x = 0; x < src_.size(); ++x)
                    {
                        filter_->apply(src_[b], dst_[b], stream_);
                    }
                }

                static const char *name() 
                {
                    return "OpenCV";
                }
            private:
                cv::Ptr<cv::cuda::Filter> filter_;
                std::vector<cv::cuda::GpuMat> src_;
                std::vector<cv::cuda::GpuMat> dst_;
                cv::cuda::Stream stream_;

                static int to_opencv(BorderMode m)
                {
                    switch(m)
                    {
                        case BorderMode::Zero:
                            return cv::BORDER_CONSTANT;
                        case BorderMode::Replicate:
                            return cv::BORDER_REPLICATE;
                        case BorderMode::Reflect:
                            return cv::BORDER_REFLECT;
                    }
                    return cv::BORDER_CONSTANT;
                }
        };
    }
}

#endif
