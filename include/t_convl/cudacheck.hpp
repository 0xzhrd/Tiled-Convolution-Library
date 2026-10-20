#pragma once

#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>

#define T_CONVL_CHECK_CUDA(call)                                                    \
    do                                                                              \
    {                                                                               \
        const cudaError_t status__ = (call);                                        \
        if(status__ != cudaSuccess)                                                 \
        {                                                                           \
            std::ostringstream oss__;                                               \
            oss__ << "CUDA error at " << __FILE__ << ':' << __LINE__ << " in '"     \
            << #call << "': " << cudaGetErrorString(status__) << "(code "           \
            << static_cast<int>(status__) << ')';                                   \
            throw std::runtime_error(oss__.str());                                  \
        }                                                                           \
    } while(0)                                                              
