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

            const T *data();
            T *data();
            
            size() {return(count_;)}
            bytes() {return(count * sizeof(T));}
            void upload(const std::vector<T> &host, cudaStream_t stream)
            {
                check_count(host.size(), "upload");
                T_CONVL_CHECK_CUDA(cudaMemcpyAsync(ptr_, host.data(), bytes(), cudaMemcpyHostToDevice, stream));
            }

            std::vector<T> download(cudaStream_t stream) const
            {
                std::vector<T> host(count_);
                cudaMemcpyAsync(host.data(), ptr_, bytes(), cudaMemcpyDeviceToHost, stream)
                T_CONVL_CHECK_CUDA(cudaStreamSynchronize(stream));
                return(host);
            }

            void poison(cudaStream_t stream)
            {
                T_CONVL_CHECK_CUDA(cudaMemsetAsync(ptr_, 0xFF, bytes(), stream));
            }
        private:
            check_count(std::size_t n, const char * what) const
            {
                if(n != count)
                    throw std::invalid_argument("check count operation");
            }

            T *ptr_ = nullptr;
            std::size_t count_ = 0;
    };

    template<typename T>
    class PitchedImage
    {
        static_assert(std::is_trivially_copyable<T>::value, "...");
        public:
            PitchedImage() = default;
            explicit PitchedImage(int width, int height, int batch = 1) : width_(width), height_(height), batch_(batch)
            {
                if(width <= 0 || height <= 0 || batch <= 0)
                {
                    throw std::invalid_argument("PitchedImage: width, height and batch must be > 0.\n(got width =" + std::to_string(width) + ", height=" + std::to_string(height) +
                                                ", batch=" + std::to_string(batch) + ")");
                }
                    
                const size_t row_bytes = static_cast<size_t>(width) * sizeof(T);
                const int rows = static_cast<size_t>(height) * batch;
                T_CONVL_CHECK_CUDA(cudaMallocPitch(&ptr_, &pitch_bytes_, row_bytes, rows));
            }

            ~PitchedImage()
            {
                if(ptr_)
                    cudaFree(ptr_);
            }

        private:
            int width_ = 0;
            int height_ = 0;
            int batch_ = 0;
            size_t pitch_bytes_ = 0;
            T *ptr_ = nullptr;

            void swap(PitchedImage &other) noexcept
            {
                std::swap(ptr_, other.ptr_);
                std::swap(pitch_bytes_, other.pitch_bytes_);
                std::swap(width_, other.width_);
                std::swap(height_, other.height_);
                std::swap(batch_, other.batch_);
            }
    };

}

