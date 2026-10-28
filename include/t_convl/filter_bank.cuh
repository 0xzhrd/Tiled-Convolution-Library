/*
this file defines the filter coefficients in constant memory through a validated host object 
and a stream ordered upload
also defines filter loading logic before kernel launch 
runs on the gou
*/

#pragma once

#ifndef __CUDACC__
#error "filter_bank.cuh declares __constant__ memory and must be compiled by nvcc"
#endif

#include <cuda_runtime.h>
#include "cudacheck.hpp"
#include "config.hpp"
#include <stdexcept>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

namespace tiled_convolution
{
    namespace detail{static __constant__ float filter_bank[filter_slots][max_filter_taps];}

    __device__ __forceinline__ float filter_tap(int slot, int index)
    {
        return detail::filter_bank[slot][index];
    }

    class Filter    //validated host side filter
    {
        public: 
            Filter(std::string name, int radius, std::vector<float> taps) : name_(std::move(name)), radius_(radius), taps_(std::move(taps))
            {
                if(name_.empty()) throw std::invalid_argument("filter name must not be empty");

                if(radius_ <= 0 || radius_ > max_filter_radius) throw std::invalid_argument("invalid filter radius");

                const std::size_t expected = static_cast<std::size_t>(width()) * width();
                if(taps_.size() != expected) throw std::invalid_argument(" ");

                for(size_t i = 0; i < taps_.size(); ++i)
                {
                    if(!std::isfinite(taps_[i])) throw std::invalid_argument("");
                }
            }

            const std::string &name() const {return name_;}
            int radius() const {return radius_;}
            int width() const {return 2 * radius_ + 1;}
            std::size_t size() const {return taps_.size();}
            const std::vector<float> &taps() const {return taps_;}
            const float *data() const {return taps_.data();}
        
        private:
            std::string name_;
            int radius_;
            std::vector<float> taps_;
    };

}
