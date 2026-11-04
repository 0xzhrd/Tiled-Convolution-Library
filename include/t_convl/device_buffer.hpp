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

            T *data() {return ptr_;}
            const T *data() {return ptr_;}
            std::size_t size() {return(count_;)}
            std::size_t bytes() {return(count * sizeof(T));}
            void upload(const std::vector<T> &host, cudaStream_t stream)
            {
                check_count(host.size(), "upload");
                T_CONVL_CHECK_CUDA(cudaMemcpyAsync(ptr_, host.data(), bytes(), cudaMemcpyHostToDevice, stream));
            }

            std::vector<T> download(cudaStream_t stream) const
            {
                std::vector<T> host(count_);
                T_CONVL_CHECK_CUDA(cudaMemcpyAsync(host.data(), ptr_, bytes(), cudaMemcpyDeviceToHost, stream));
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

            PitchedImage(const PitchedImage &) = delete;
            PitchedImage &operator = (const PitchedImage &) = delete;

            PitchedImage(PitchedImage&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)), pitch_bytes_(std::exchange(other.pitch_bytes_, 0)), width_(std::exchange(other.width_, 0)), height_(std::exchange(other.height_, 0)), batch_(std::exchange(other.batch_, 0))
            { }

            PitchedImage &operator = (PitchedImage&& other) noexcept
            {
                if(this != &other)
                {
                    cudaFree(ptr_);
                    ptr_ = std::exchange(other.ptr_, nullptr);
                    pitch_bytes_ = std::exchange(other.pitch_bytes_, 0);
                    width_ = std::exchange(other.width_, 0);
                    height_ = std::exchange(other.height_, 0);
                    batch_ = std::exchange(other.batch_, 0);
                }
                return *this;
            }

            int width() const {return width_;}
            int height() const {return height_;}
            int batch() const {return batch_;}
            size_t pitch_bytes() const {return pitch_bytes_;}

            std::ptrdiff_t image_stride() const
            {
                return static_cast<std::ptrdiff_t>(pitch_bytes_ / sizeof(T)) * height_;
            }

            ImageView<T> view()
            {
                return make_view(ptr_, width_, height_, pitch_bytes_);
            }

            ImageView<const T> const_view() const
            {
                return make_view<const T>(ptr_, width_, height_, pitch_bytes_);
            }

            



        private:
            std::size_t total_rows() const
            {
                return static_cast<std::size_t>(height_) * static_cast<std::size_t>(batch_);
            }

            std::size_t element_count() const 
            {
                return static_cast<std::size_t>(width_) * total_rows(); 
            }

            void check_count(std::size_t n, const char *what) const
            {
                if(n != element_count())
                {
                    throw std::invalid_argument(std::string("PitchedImage:") + what + ": host has " + std::to_string(n) + " elements, image needs " +
                                                std::to_string(element_count()));
                }
            }
            
            size_t pitch_bytes_ = 0;
            T *ptr_ = nullptr;
            int width_ = 0;
            int height_ = 0;
            int batch_ = 0;

    };

}

