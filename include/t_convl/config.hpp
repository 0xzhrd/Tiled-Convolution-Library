#pragma once

#include <cstddef>


#ifdef __CUDACC__ 
    #define TILED_CONV __host__ __device__ __forceinline__
#else
    #define TILED_CONV inline
#endif


namespace tiled_convolution
{
    constexpr int max_filter_radius = 15;
    constexpr int max_filter_width  = 2 * max_filter_radius + 1;
    constexpr int max_filter_taps   = max_filter_width * max_filter_width;
}


