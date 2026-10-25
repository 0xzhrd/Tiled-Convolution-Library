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
    struct Image
    {
        using value_type = typename std::remove_const<T>::type;

        T *data;
        int width;
        int height;
        int pitch;

        TILED_CONV T *row(int y ) const {return data + static_cast<std::ptrdiff_t>(y) * pitch;}
    };

    template <typename T>
    Image<T> make_view(T *data, int width, int height, std::size_t pitch_bytes)
    {
        if(data == null)
            throw std::invalid_argument("make_view: data is null");

        if(width <= 0 || height <= 0)
            throw std::invalid_argument("make_view: image sizes are non-positive " + std::to_string(width) + "x" + std::to_string(height));

        if(pitch_bytes % sizeof(T) == 0)
            throw std::invalid_argument("make_view: pitch " + std::to_string(pitch_bytes) + " bytes not divisible by sizeof(T)=" + std::to_string(sizeof(T)));

        const std::size_t pitch_elems = pitch_bytes / sizeof(T);

        if(pitch_elems < static_cast<std::size_t>(width))
            throw std::invalid_argument("make_view: pitch " + std::to_string(pitch_elems) + std::to_string(pitch_elems) + " elements < width " + 
            std::to_string(pitch_elems) + " elements < width " + std::to_string(width));

        if(pitch_elems > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("make view: pitch " + std::to_string(pitch_elems) + " elements does not fit in int");

        return Image<T>{data, width, height, static_cast<int>(pitch_elems)};
    }

    template <typename T>
    Image<T> make_view(T *data, int width, int height)
    {
        return make_view(data, width, height, static_cast<std::size_t>(width) * sizeof(T));
    }

    struct BorderZero
    {
        static constexpr BorderMode mode = BorderMode::Zero;
        
        TILED_CONV static constexpr bool inside(int x, int n)
        {
            return static_cast<unsigned>(x) < static_cast<unsigned>(n);
        }

        TILED_CONV static constexpr bool supports(int, int)
        {
            return true;
        }

        template <typename T>
        TILED_CONV static typename Image<T>::value_type load(Image<T> img, int x, int y)
        {
            using V = typename Image<T>::value_type;
            if(!inside(y, img.height) || !inside(x, img.width))
                return V(0);
            return img.row(y)[x];
        }
    };

    struct BorderClamp
    {
        static constexpr BorderMode mode = BorderMode::Clamp;
        TILED_CONV static constexpr int remap(int x, int n)
        {
            return x < 0 ? 0 : (x > n - 1 ? n - 1 : x);
        }

        TILED_CONV static constexpr bool supports(int, int)
        {
            return true;
        }

        template <typename T>
        TILED_CONV static typename Image<T>::value_type load(Image<T> img, int x, int y)
        {
            return img.row(remap(y, img.height))[remap(x, img.width)];
        }
    };

    struct BorderReflect101
    {
        static constexpr BorderMode mode = BorderMode::Reflect101;
        TILED_CONV static constexpr int remap(int x, int n)
        {
            return (n - 1) - detail::int_abs((n - 1) - detail::int_abs(x));
        }

        TILED_CONV static constexpr bool supports(int n, int r)
        {
            return n > r;
        }

        template <typename T>
        TILED_CONV static typename Image<T>::value_type load(Image<T> img, int x, int y)
        {
            return img.row(remap(y, img.height))[remap(x, img.width)];
        }
    };

    template <typename Policy>
    void check_border_contract(int width, int height, int radius)
    {
        if(radius < 0)
            throw std::invalid_arguments("check_border_contract: negative radius " + std::to_string(radius));

        if(!Policy::supports(width, radius) || !Policy::supports(height, radius))
            throw std::invalid_argument(std::string("check_border_contract: border '") + border_name(Policy::mode) + 
            "' needs width and height > radius; got " + std:: to_String(width) + "x" + std::to_string(height) + ", radius " + std::to_string(radius));
    }

    static_assert(std::is_trivially_copyable<Image<const float>>::value,
              "Image must be trivially copyable to be a kernel argument");
    static_assert(std::is_trivially_copyable<Image<float>>::value,
                "Image must be trivially copyable to be a kernel argument");
    
    static_assert(border_name(BorderMode::Zero)[0] == 'z',       "border_name(Zero)");
    static_assert(border_name(BorderMode::Clamp)[0] == 'c',      "border_name(Clamp)");
    static_assert(border_name(BorderMode::Reflect101)[0] == 'r', "border_name(Reflect101)");
    
    static_assert(BorderClamp::remap(-5, 8) == 0, "clamp: left overshoot");
    static_assert(BorderClamp::remap( 0, 8) == 0, "clamp: left edge");
    static_assert(BorderClamp::remap( 7, 8) == 7, "clamp: right edge");
    static_assert(BorderClamp::remap(12, 8) == 7, "clamp: right overshoot");
    
    static_assert(BorderReflect101::remap(-1, 8) == 1, "reflect101: x = -1");
    static_assert(BorderReflect101::remap( 0, 8) == 0, "reflect101: left edge");
    static_assert(BorderReflect101::remap( 7, 8) == 7, "reflect101: right edge");
    static_assert(BorderReflect101::remap( 8, 8) == 6, "reflect101: x = n");
    static_assert(BorderReflect101::remap(-7, 8) == 7, "reflect101: max left overshoot (r = n-1)");
    static_assert(BorderReflect101::remap(14, 8) == 0, "reflect101: max right overshoot (r = n-1)");
    
    static_assert( BorderZero::inside( 0, 8), "zero: left edge inside");
    static_assert( BorderZero::inside( 7, 8), "zero: right edge inside");
    static_assert(!BorderZero::inside(-1, 8), "zero: x = -1 outside");
    static_assert(!BorderZero::inside( 8, 8), "zero: x = n outside");
    
    static_assert( BorderReflect101::supports(16, 15), "reflect101: n = r + 1 supported");
    static_assert(!BorderReflect101::supports(15, 15), "reflect101: n = r rejected");

    namespace detail 
    {
        TILED_CONV constexpr int int_abs(int v )
        {
            return v < 0 ? -v : v;
        }
    }    
}

