/*

*/

#include "t_convl/cudacheck.hpp"
#include "t_convl/filter_bank.cuh"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

using namespace tiled_convolution;

namespace
{
    char g_ctx[200];
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

    template <typename E, typename F>
    void expect_throw_as(F lambda, const char *what, int line)
    {
        ++g_checks;
        try
        {
            lambda();
        }
        catch(const E&)
        {
            return;
        }
        catch(const std::exception &e)
        {
            ++g_failures;
            std::printf("FAIL line %d: %s threw the wrong exception type: %s\n", line, what, e.what());
            return;
        }
        catch(...)
        {
            ++g_failures;
            std::printf("FAIL line %d: %s threw a non-std exception\n", line, what);
            return;
        }
        ++g_failures;
        std::printf("FAIL line %d :%s did not throw\n", line, what);
    }

    #define EXPECT_THROW_AS(Type, expr) expect_throw_as<Type>([&] {(void)(expr);}, #expr, __LINE__)
}