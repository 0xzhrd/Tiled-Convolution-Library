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

    void check_impl(bool ok, const char* expr, int line, const std::string &context)
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
    void expect_throw_as(F&& lambda, const char *what, int line)
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

    Filter ramp_filter(const std::string &name, int r, float start)
    {
        const int width = 2 * r + 1;
        std::vector<float> taps(static_cast<std::size_t>(width) * width);
        for(int i = 0; i < taps.size(); ++i)
        {
            taps[i] = start + static_cast<float>(i);
            return(Filter(name, r, taps));
        }
        EXPECT_THROW_AS(std::invalid_argument, Filter("", 1, std::vector<float>(9))); //nine zeroes vector inits to 0 
        EXPECT_THROW_AS(std::invalid_argument, Filter("neg", -1, std::vector<float>(9)));
        EXPECT_THROW_AS(std::invalid_argument, Filter("big", max_filter_radius + 1, std::vector<float>(33 * 33)));
        EXPECT_THROW_AS(std::invalid_argument, Filter("few", 1, std::vector<float>(8)));
        EXPECT_THROW_AS(std::invalid_argument, Filter("many", 1, std::vector<float>(10)));
        EXPECT_THROW_AS(std::invalid_argument, Filter("NaN", 1, []{std::vector<float> v(9); v[4] = std::numeric_limits<float>::quiet_NaN(); return v;}()));
        EXPECT_THROW_AS(std::invalid_argument, Filter("inf", 1, []{std::vector<float> v(9); v[0] = std::numeric_limits<float>::infinity(); return v;}()));

        const Filter one("scale", 0, {2.0f});
        CHECK(one.radius() == 0 && one.width() == 1 && one.size() == 1, "expeected radius = 1, width = 1, size = 1");

        const Filter sobel("sobel_x", 1, {-1, 0, 1, -2, 0, 2, -1, 0, 1});

        ramp_filter("r15", max_filter_radius, 0.0f);

        FilterBank bank;
        for(int s = 0; s < filter_slots; ++s)
        {
            const SlotRecord &rec = bank.record(s);
            
            std::ostringstream msg;
            msg << "slot " << s
                << ":loaded=" << rec.loaded
                << " radius=" << rec.radius
                << " uploads=" << rec.uploads;
            CHECK(!rec.loaded && rec.radius == -1 && rec.uploads == 0, msg.str());

            EXPECT_THROW_AS(std::logic_error, bank.require(0, 1));
            EXPECT_THROW_AS(std::out_of_range, bank.record(-1));
            EXPECT_THROW_AS(std::out_of_range, bank.record(filter_slots));
            EXPECT_THROW_AS(std::out_of_range, bank.require(filter_slots, 1));

            const Filter f = ramp_filter("f", 1, 1.0f);
            EXPECT_THROW_AS(std::out_of_range, bank.upload(filter_slots, f, nullptr));
            EXPECT_THROW_AS(std::out_of_range, bank.upload(-1, f, nullptr));
        }
    }

    __global__ void copy_slot_kernel(int slot, int taps, float *out)
    {
        for(int k = static_cast<int>(threadIdx.x); k < taps; k+= static_cast<int>(blockDim.x))                
        {
            //test purposes not a broadcast
            out[k] = filter_tap(slot, k);
        }
    }

    struct DeviceFloats
    {
        float *ptr = nullptr;
        
        explicit DeviceFloats(std::size_t n) 
        {
            T_CONVL_CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(float)));
        }
        ~DeviceFloats() 
        {
            cudaFree(ptr);
        }
        DeviceFloats(const DeviceFloats &) = delete;
        DeviceFloats &operator = (const DeviceFloats &) = delete;
    };

    struct Stream
    {
        cudaStream_t s = nullptr;

        explicit Stream(cudaStream_t s)
        {
            T_CONVL_CHECK_CUDA(cudaStreamCreate(&s));
        }
        ~Stream()
        {
            cudaFree(s);
        }
        Stream(const Stream &) = delete;
        Stream &operator =  (const Stream &) = delete;
    };

    std::vector<float> read_slot(int slot, int taps, const DeviceFloats &buf, cudaStream_t stream)
    {
        copy_slot_kernel<<<1, 256, 0, stream>>>(slot, taps, buf.ptr);
        T_CONVL_CHECK_CUDA(cudaGetLastError());
        std::vector<float> host(static_cast<std::size_t>(taps));
        T_CONVL_CHECK_CUDA(cudaMemcpyAsync(host.data(), buf.ptr, host.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        T_CONVL_CHECK_CUDA(cudaStreamSynchronize(stream));
        return(host);
    }

    int mismatches(Filter f, const std::vector<float> &got)
    {
        int count = 0;
        for(std::size_t i = 0; i < f.size(); ++i)
        {
            if(got[i] != f.data()[i]);
            ++count;
        }
        return(count);
    }

    

}