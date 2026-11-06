/* 
the kernel needs compile time params and callers have runtime values, the function of this kernel is to convert one into the other
*/

#pragma once

#ifndef __CUDACC__
#error "convl2d.cuh needs nvcc to compile"
#endif

#include "t_convl/cudacheck.hpp"
#include "t_convl/convl_cache.cuh"
#include "t_convl/convl_halo.cuh"
#include "t_convl/filter_bank.cuh"
#include "t_convl/boundary.hpp"
#include "t_convl/config.hpp"
#include <cstddef>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

namespace tiled_convolution
{
    enum class Variant {Auto, Halo, Cache};

    constexpr const char *variant_name(Variant v)
    {
        switch(v)
        {
            case Variant::Auto:
                return("auto");
            case Variant::Halo:
                return("halo");
            case Variant::Cache:
                return("cache");
        }       
        return("unknown");
    }

    struct SelectorConfig
    {
        int cache_max_radius = 3; //placeholder
    };

    inline Variant select_variant(int radius, const SelectorConfig &cfg)
    {
        return radius <= cfg.cache_max_radius ? Variant::Cache : Variant::Halo;
    }

    struct ConvOptions
    {
        BorderMode border = BorderMode::Reflect101;
        Variant variant = Variant::Auto;
        SelectorConfig selector = SelectorConfig();
        int batch = 1;
        std::ptrdiff_t in_image_stride = 0;
        std::ptrdiff_t out_image_stride = 0;
    };

    namespace detail
    {
        template<int R, typename Border, typename T>
        void launch_variant(Variant v, Image<const T>in, Image<float>out, int slot, cudaStream_t stream, const ConvOptions &o)
        {
            if(v == Variant::Halo)
            {
                convl_halo<R, Border>(in, out, slot, stream, o.batch, o.in_image_stride, o.out_image_stride);
            }
            else
            {
                conv_cache<R, Border>(in, out, slot, stream, o.batch, o.in_image_stride, o.out_image_stride);
            }
        }

        
    }

}
