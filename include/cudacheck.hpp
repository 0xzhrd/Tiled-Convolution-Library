#pragma once

#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do                                                                      \
    {                                                                       \
        const cudaError_t status__ = (call);                                \
        if(status__ != cudaSuccess)                                         \
        {                                                                   \
            std::ostringstream oss__;                                       \
            oss__ << "CUDA error at " << FILE << ':' << LINE << " in '"     \
            << #call << "': " << cudaGetErrorString(status__) << "(code "   \
            << static_cast<int>(status__) << ')';                           \
            throw std::runtime_error(oss__.str());                          \   
        }                                                                   \
    } while(0)                                                              \
