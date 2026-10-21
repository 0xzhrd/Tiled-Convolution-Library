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
                return "zero";

            case BorderMode::Clamp:
                return "clamp";

            case BorderMode::Reflect101:
                return "reflext101";
        }
        return "unknown";
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

    template <typename T>
    ImageView<T> make_view(T *data, int width, int height, std::size_t pitch_bytes)
    {
        if(data == null)
            throw std::invalid_argument("make_view: data is null");

        if(width <= 0 || height <= 0)
            throw std::invalid_argument("make_view: image sizes are non-positive " + std::to_string(width) + "x" + std::to_string(height));

        if(pitch_bytes % sizeof(T) == 0)
            throw std::invalid_argument("make_view: pitch " + std::to_string(pitch_bytes) + " bytes not divisible by sizeof(T)=" + std::to_string(sizeof(T)));

        
        
    }

    template <typename T>
    ImageView<T> make_view(T *data, int width, int height);

    struct BorderZero
    {
        static constexpr BorderMode mode = BorderMode::Zero;
        TILED_CONV static constexpr bool supports(int n, int r);
        template <typename T>
        TILED_CONV static typename ImageView<T>::value_type load(ImageView<T> img, int x, int y);
    };

    struct BorderClamp
    {
        static constexpr BorderMode mode = BorderMode::Clamp;
        TILED_CONV static constexpr bool supports(int n, int r);
        template <typename T>
        TILED_CONV static typename ImageView<T>::value_type load(ImageView<T> img, int x, int y);
    };

    struct BorderReflect101
    {
        static constexpr BorderMode mode = BorderMode::Reflect101;
        TILED_CONV static constexpr bool supports(int n, int r);
        template <typename T>
        TILED_CONV static typename ImageView<T>::value_type load(ImageView<T> img, int x, int y);
    };

    namespace detail 
    {
        TILED_CONV constexpr int iabs(int v )
        {
            return v < 0 ? -v : v;
        }
    }    
}

