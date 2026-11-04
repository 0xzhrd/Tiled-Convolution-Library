/*

*/

#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <utility>
#include <type_traits>
#include <cuda_runtime.h>
#include "t_convl/boundary.hpp"
#include "t_convl/cudacheck.hpp"

namespace tiled_convolution
{
    template <typename T>
    class DeviceBuffer
    {
        static_assert(std::is_trivially_copyable<T>::value, "...");
        public:
            DeviceBuffer() = default;

            explicit DeviceBuffer(std::size_t count) : count_(count)
            {
                if(count_ > 0)
                    T_CONVL_CHECK_CUDA(cudaMalloc(&ptr_, bytes()));
            }

            ~DeviceBuffer() {cudaFree(ptr_);}

            DeviceBuffer(const DeviceBuffer &) = delete;
            DeviceBuffer &operator = (const DeviceBuffer &) = delete;

            DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)), count_(std::exchange(other.count_, 0))
            { }

            DeviceBuffer &operator = (DeviceBuffer&& other) noexcept
            {
                if(this != &other)
                {
                    cudaFree(ptr_);
                    ptr_ = std::exchange(other.ptr_, nullptr);
                    count_ = std::Exchange(other.count_, 0);
                }
                return *this;
            }

            
    };
}

