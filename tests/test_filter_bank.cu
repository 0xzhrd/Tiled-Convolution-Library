/*

*/

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
        }
        return(Filter(name, r, taps));
    }

    void _test_filter_validation()
    {
        EXPECT_THROW_AS(std::invalid_argument, Filter("", 1, std::vector<float>(9))); //nine zeroes vector inits to 0 
        EXPECT_THROW_AS(std::invalid_argument, Filter("neg", -1, std::vector<float>(9)));
        EXPECT_THROW_AS(std::invalid_argument, Filter("big", max_filter_radius + 1, std::vector<float>(33 * 33)));
        EXPECT_THROW_AS(std::invalid_argument, Filter("few", 1, std::vector<float>(8)));
        EXPECT_THROW_AS(std::invalid_argument, Filter("many", 1, std::vector<float>(10)));

        EXPECT_THROW_AS(std::invalid_argument, Filter("NaN", 1, []{std::vector<float> v(9); v[4] = std::numeric_limits<float>::quiet_NaN(); return v;}()));
        EXPECT_THROW_AS(std::invalid_argument, Filter("inf", 1, []{std::vector<float> v(9); v[0] = std::numeric_limits<float>::infinity(); return v;}()));
    }

    void test_filter_accessors()
    {
        const Filter one("scale", 0, {2.0f});
        CHECK(one.radius() == 0 && one.width() == 1 && one.size() == 1, "expeected radius = 1, width = 1, size = 1");

        const Filter sobel("sobel_x", 1, {-1, 0, 1, -2, 0, 2, -1, 0, 1});

        ramp_filter("r15", max_filter_radius, 0.0f);
    }

    void test_bank_bookkeeping()
    {
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
        }

        EXPECT_THROW_AS(std::logic_error, bank.require(0, 1));
        EXPECT_THROW_AS(std::out_of_range, bank.record(-1));
        EXPECT_THROW_AS(std::out_of_range, bank.record(filter_slots));
        EXPECT_THROW_AS(std::out_of_range, bank.require(filter_slots, 1));

        const Filter f = ramp_filter("f", 1, 1.0f);
        EXPECT_THROW_AS(std::out_of_range, bank.upload(filter_slots, f, nullptr));
        EXPECT_THROW_AS(std::out_of_range, bank.upload(-1, f, nullptr));
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

    int mismatches(const std::vector<float> &got, const Filter &f)
    {
        int count = 0;
        for(std::size_t i = 0; i < f.size(); ++i)
        {
            if(got[i] != f.data()[i])
            ++count;
        }
        return(count);
    }

    void test_gpu_round_trip()
    {
        FilterBank bank;
        Stream stream;
        DeviceFloats buf(static_cast<std::size_t>(max_filter_taps));

        const Filter a = ramp_filter("a_r1", 1, 1.0f);
        const Filter b = ramp_filter("b_r3", 3, 100.0f);
        const Filter c = ramp_filter("c_r15", max_filter_radius, 1000.0f);

        bank.upload(2, a, stream.s);
        bank.upload(0, b, stream.s);
        bank.upload(filter_slots - 1, c, stream.s);

        int bad = mismatches(read_slot(2, static_cast<int>(a.size()), buf, stream.s), a);
        std::snprintf(g_ctx, sizeof(g_ctx), "slot 2 (r1): %d coefficients differ", bad);
        CHECK(bad == 0, g_ctx);

        bad = mismatches(read_slot(0, static_cast<int>(b.size()), buf, stream.s), b);
        std::snprintf(g_ctx, sizeof(g_ctx), "slot 0 (r3): %d coefficients differ", bad);
        CHECK(bad == 0, g_ctx);

        bad = mismatches(read_slot(filter_slots - 1, static_cast<int>(c.size()), buf, stream.s), c);
        std::snprintf(g_ctx, sizeof(g_ctx), "last slot (r15, full): %d coefficients differ", bad);
        CHECK(bad == 0, g_ctx);

        const Filter d = ramp_filter("d_r15", max_filter_radius, -500.0f);
        bank.upload(3, d, stream.s);
        bad = mismatches(read_slot(2, static_cast<int>(a.size()), buf, stream.s), a);
        std::snprintf(g_ctx, sizeof(g_ctx), "slot f2 after writing slot 3: %d coefficients differ", bad);
        CHECK(bad == 0, g_ctx);

        CHECK(bank.record(2).loaded && bank.record(2).radius == 1 && bank.record(2).name == "a_r1", "slot 2 record");
        CHECK(bank.record(2).uploads == 1, "slot 2 uploaded once");

        bool ok = true;
        try
        {
            bank.require(2, 1);
            bank.require(filter_slots - 1, max_filter_radius);
        }
        catch(const std::exception& e)
        {
            ok = false;
            std::printf(" require threw: %s\n", e.what());
        }
        CHECK(ok, "require with matching radius passes");
        EXPECT_THROW_AS(std::logic_error, bank.require(2, 3));
        EXPECT_THROW_AS(std::logic_error, bank.require(1,1 ));

        const Filter a2 = ramp_filter("a2_a1", 1, 50.0f);
        bank.upload(2, a2, stream.s);
        bad = mismatches(read_slot(2, static_cast<int>(a.size()), buf, stream.s), a2);
        std::snprintf(g_ctx, sizeof(g_ctx), "slot 2 after re-upload: %d coefficients differ", bad);
        CHECK(bad == 0, g_ctx);
        CHECK(bank.record(2).uploads == 2 && bank.record(2).name == "a2_a1", "re-upload recorded");
    }

    void run(void (*test)(), const char *name)
    {
        try
        {
            test();
        }
        catch(const std::exception& e)
        {
            ++g_failures;
            std::printf("FAIL: %s aborted with an unexpected exception: %s\n", name, e.what());
        }
    }

    bool gpu_available()
    {
        int count = 0;
        const cudaError_t err = cudaGetDeviceCount(&count);
        if(err != cudaSuccess)
        {
            std::printf("no usable GPU: %s\n", cudaGetErrorString(err));
            cudaGetLastError();
            return(false);
        }
        return(count > 0);
    }
}

int main()
{
    run(test_filter_validation, "test_filter_validation");
    run(test_filter_accessors, "test_filter_accessors");
    run(test_bank_bookkeeping, "test_bank_bookeeping");

    if(!gpu_available())
    {
        std::printf("filter_bank: %d failures in %d host checks; GPU checks SKIPPED\n", g_failures, g_checks);
        return(g_failures == 0 ? 77 : 1);
    }

    run(test_gpu_round_trip, "test_gpu_rounf_trip");
    std::printf("filtr_bank: %d failures in %d checks\n", g_failures, g_checks);
    return(g_failures == 0 ? 0 : 1);
}