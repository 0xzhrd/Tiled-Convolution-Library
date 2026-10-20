#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "config.hpp"

namespace tiled_convolution
{
    enum class BorderMode {Zero, Clamp, Reflect101};

    constexpr const char *border_name(BorderMode m) 
    {
        switch (m)
        {
            case BorderMode::Zero:
                break;

            case BorderMode::Clamp:
                break;

            case BorderMode::Reflect101:
                break;
        }
        return "Unknown";
    }

    template <typename T>
    struct ImageView
    {
        using value_type = typename std::remove_const<T>::type;

        T *data;
        int width;
        int height;
        int pitch;

        TILED_CONV T *row(int y ) const {return data + static_cast<std::ptrdiff_t>(y) * pitch;}
    };

    namespace detail 
    {
        TILED_CONV constexpr int iabs(int v )
        {
            return v < 0 ? -v : v;
        }
    }    
}

