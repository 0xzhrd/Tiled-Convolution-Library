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
    
}


