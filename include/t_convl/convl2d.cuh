/* 
the kernel needs compile time params and callers have runtime values, the function of this kernel is to convert one into the other
*/

#pragma once

#ifndef __CUDACC__
#error "convl2d.cuh needs nvcc to compile"
#endif

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
        void launch_variant(Variant v, Image<const T> in, Image<float> out, int slot, cudaStream_t stream, const ConvOptions &o)
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

        template<int R, typename T>
        void launch_border(Variant v, Image<const T> in,  Image<float> out, int slot, cudaStream_t stream, const ConvOptions &o)
        {
            switch(o.border)
            {
                case BorderMode::Zero:
                    launch_variant<R, BorderZero>(v, in, out, slot, stream, o);
                    return;
                case BorderMode::Clamp: 
                    launch_variant<R, BorderClamp>(v, in, out, slot, stream, o);
                    return;
                case BorderMode::Reflect101: 
                    launch_variant<R, BorderReflect101>(v, in, out, slot, stream, o);
                    return;
            }
            throw std::invalid_argument("convl2d: unknown border mode (this happens if you pass an out of range value into the enum)");
        }

        static_assert(max_filter_radius == 15, "convl2d's radius switch covers 0 to 15; update it to match config.hpp");

        template<typename T>
        void launch_radius(int radius, Variant v, Image<const T> in, Image<float> out, cudaStream_t stream, const ConvOptions &o)
        {
            switch(radius)
            {//if build time is an issue, make the switch case for the radii narrower
                case 0:
                    launch_border<0>(v, in, out, slot, stream, o);
                    return;
                case 1:
                    launch_border<1>(v, in, out, slot, stream, o);
                    return;
                case 2:
                    launch_border<2>(v, in, out, slot, stream, o);
                    return;
                case 3:
                    launch_border<3>(v, in, out, slot, stream, o);
                    return;
                case 4:
                    launch_border<4>(v, in, out, slot, stream, o);
                    return;
                case 5:
                    launch_border<5>(v, in, out, slot, stream, o);
                    return;
                case 6:
                    launch_border<6>(v, in, out, slot, stream, o);
                    return;
                case 7:
                    launch_border<7>(v, in, out, slot, stream, o);
                    return;
                case 8:
                    launch_border<8>(v, in, out, slot, stream, o);
                    return;
                case 9:
                    launch_border<9>(v, in, out, slot, stream, o);
                    return;
                case 10:
                    launch_border<10>(v, in, out, slot, stream, o);
                    return;
                case 11:
                    launch_border<11>(v, in, out, slot, stream, o);
                    return;
                case 12:
                    launch_border<12>(v, in, out, slot, stream, o);
                    return;
                case 13:
                    launch_border<13>(v, in, out, slot, stream, o);
                    return;
                case 14:
                    launch_border<14>(v, in, out, slot, stream, o);
                    return;
                case 15:
                    launch_border<15>(v, in, out, slot, stream, o);
                    return;
                default:
                    break;
            }
            throw std::invalid_argument("convl2d: radius " + std::to_string(radius) + " outside [0, " + std::to_string(max_filter_radius) + "]");
        }
    }

    template <typename T>
    Variant convl2d(const FilterBank &bank, int slot, int radius, Image<const T> in, Image<float> out, cudaStream_t stream, const ConvOptions &options = ConvOptions())
    {
        if(radius < 0 || radius > max_filter_radius)
        {
            throw std::invalid_argument("convl2d: radius outside [0, " + std::to_string(max_filter_radius) + "]");
        }
        bank.require(slot, radius);
        const Variant chosen = options.variant == Variant::Auto ? select_variant(radius, options.selector)

        detail::launch_radius(radius, chosen, in, out, slot, stream, options);
        return chosen;
    }
}
