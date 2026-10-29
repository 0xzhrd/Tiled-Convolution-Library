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

    struct SlotRecord
    {
        bool loaded; //checks if naything has been uploaded
        int radius; //radius of the filter in the slot
        std::string name; //filter name
        unsigned long uploads; //how many times has the slot been written
    };

    class FilterBank
    {
        public:
            FilterBank() : records_(static_cast<std::size_t>(filter_slots), SlotRecord{false, - 1, std::string(), 0}) {}

            FilterBank(const FilterBank &) = delete;
            FilterBank &operator = (const FilterBank &) = delete;

            void upload(int slot, const Filter &filter, cudaStream_t stream)
            {
                check_slot(slot);
                const std::size_t offset = static_cast<std::size_t>(slot) * max_filter_taps * sizeof(float);
                const std::size_t bytes = filter.size() * sizeof(float);
                T_CONVL_CHECK_CUDA(cudaMemcpyToSymbolAsync(detail::filter_bank, filter.data(), bytes, offset, cudaMemcpyHostToDevice, stream));
                SlotRecord &rec = records_[static_cast<std::size_t>(slot)];
                rec.loaded = true;
                rec.radius = filter.radius();
                rec.name = filter.name();
                ++rec.uploads;
            }

            void require(int slot, int radius) const
            {
                check_slot(slot);
                const SlotRecord &rec = records_[static_cast<std::size_t>(slot)];

                if(!rec.loaded)
                    throw std::logic_error("FilterBank: slot " + std::to_string(slot) + "is empty");

                if(rec.radius != radius)
                    throw std::logic_error("FilterBank: slot " + std::to_string(slot) + "holds '" + rec.name + "' (radius " + std::to_string(rec.radius) +
                                            "), kernel expects radius " + std::to_string(radius));
            }

            const SlotRecord &record(int slot) const
            {
                check_slot(slot);
                return records_[static_cast<std::size_t>(slot)];
            }

    
        private:
            std::vector<SlotRecord> records_;
            void check_slot(int slot) const
            {
                if(slot < 0 || slot >= filter_slots)
                    throw std::out_of_range("FilterBank: slot " + std::to_string(slot) + " outside [0, " + std::to_string(filter_slots) +"]");
            }
    };
}
