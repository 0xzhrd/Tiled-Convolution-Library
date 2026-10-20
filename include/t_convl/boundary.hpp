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

    template <typename T>
    ImageView<T> make_view(T *data, int width, int height, std::size_t pitch_bytes);

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

