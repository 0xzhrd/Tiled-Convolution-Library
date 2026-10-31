/*

*/

#include "t_convl/reference.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <cstdint>
#include <stdexcept>
#include <cstdio>
#include <vector>

using namespace tiled_convolution;
namespace ref = tiled_convolution::reference;

namespace     //everything inside this namespace is private to this file
{
    char g_ctx[120];
    int g_checks = 0;
    int g_failures = 0;

    void check_impl(bool ok, const char* expr, int line, const char* context)
    {
        ++g_checks;
        if (!ok)
        {
            ++g_failures;
            std::printf("FAIL line %d: %s  [%s]\n", line, expr, context);
        }
    }

    // Stringize the condition and capture the line, like CHECK_CUDA does.
    #define CHECK(cond, context) check_impl((cond), #cond, __LINE__, (context))

    // Expect the callable to throw exactly std::invalid_argument.
    template <typename F>
    void expect_invalid_argument(F&& f, const char* what, int line)
    {
        ++g_checks;
        try
        {
            f();
        }
        catch (const std::invalid_argument&)
        {
            return; // pass
        }
        catch (const std::exception& e)
        {
            ++g_failures;
            std::printf("FAIL line %d: %s threw the wrong exception type: %s\n", line, what, e.what());
            return;
        }
        catch (...)
        {
            ++g_failures;
            std::printf("FAIL line %d: %s threw a non-std exception\n", line, what);
            return;
        }
        ++g_failures;
        std::printf("FAIL line %d: %s did not throw\n", line, what);
    }

    #define EXPECT_THROW(expr) expect_invalid_argument([&] { (void)(expr); }, #expr, __LINE__)

    const BorderMode kModes[] = {BorderMode::Zero, BorderMode::Clamp, BorderMode::Reflect101};
    
    constexpr int kW = 37;
    constexpr int kH = 23;

    std::vector<float> coord_image()
    {
       const size_t size = static_cast<size_t>(kW) * kH;
       std::vector<float> result(size);
       for(int y = 0; y < kH; ++y)
        {
            for(int x = 0; x < kW; ++x)
            {//row major
                result[static_cast<size_t>(y) * static_cast<size_t>(kW) + static_cast<size_t>(x)] = static_cast<float>(10 * y + x);
            }
        }
        return(result);
    }

    float at(const std::vector<float> &img, int x, int y)
    {
        return(img[y * kW + x]);
    }

    std::vector<float> delta_filter(int r, int dx, int dy)
    {
        const int fw = ref::filter_width(r);
        std::vector<float> result(static_cast<size_t>(fw) * static_cast<size_t>(fw), 0.0f);
        result[static_cast<size_t>(dy + r) * static_cast<size_t>(fw) + static_cast<size_t>(dx + r)] = 1.0f;
        return(result);
    }

    
}