/*
one exception type translating to : "this librbray does not support this configuration"
so that is is differentiable from a genuine error to the benchmark
*/

#pragma once

#include <stdexcept>
#include <string>

namespace tiled_convolution
{
    namespace baseline
    {
        class Unsupported : public std::runtime_error
        {
            public:
                explicit Unsupported(const std::string &why) : std::runtime_error(why)
                {}
        };
    }
}
