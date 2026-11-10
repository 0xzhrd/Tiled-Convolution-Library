/*
Test of the baseline wrappers: each library, on the configurations it supports, must agree
with reference::correlate. Otherwise the benchmark would be timing a library computing a
DIFFERENT result, and the comparison would be meaningless.

It also settles NPP's convention on the hardware: with the flipped upload NPP must match the
reference; with the unflipped upload it must NOT (the filter is asymmetric).

Build (OpenCV part only with -DT_CONVL_WITH_OPENCV and OpenCV's CUDA modules):
  nvcc -std=c++14 -I include -I tests -I benchmarks/baselines \
    -gencode arch=compute_87,code=sm_87 -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 tests/test_baselines.cu -o build/test_baselines \
    -lcudnn -lnppif -lnppc
Exit code: 0 = pass, 1 = failure, 77 = host checks passed but no GPU.
*/

#include "t_convl/boundary.hpp"
#include "t_convl/device_buffer.hpp"
#include "t_convl/filter_bank.cuh"
#include "t_convl/reference.hpp"

#include "common_baseline.hpp"
#include "cudnn_baseline.cuh"
#include "npp_baseline.cuh"
#include "opencv_baseline.cuh"

#include "check.hpp"

#include <cstddef>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using namespace tiled_convolution;
namespace ref = tiled_convolution::reference;

namespace
{

constexpr int kW = 61, kH = 37, kBatch = 2;

struct Context
{
    Stream       stream;
    std::mt19937 rng{99u};
};
Context* g_ctx = nullptr;

Filter random_filter(int r)
{
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int w = 2 * r + 1;
    std::vector<float> taps(static_cast<std::size_t>(w) * static_cast<std::size_t>(w));
    for (float& t : taps)
        t = dist(g_ctx->rng);
    return Filter("rand_r" + std::to_string(r), r, taps);
}

// Run one baseline on a pitched batch and compare every image with the reference.
// Returns the number of failing pixels over the batch, or -1 if the library is Unsupported.
template <typename Baseline, typename... Extra>
long compare_baseline(const Filter& f, BorderMode border, Extra... extra)
{
    const cudaStream_t s = g_ctx->stream.get();

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const std::size_t per_image = static_cast<std::size_t>(kW) * kH;
    std::vector<float> host(per_image * kBatch);
    for (float& p : host)
        p = dist(g_ctx->rng);

    PitchedImage<float> in(kW, kH, kBatch), out(kW, kH, kBatch);
    in.upload(host, s);
    out.poison(s);

    try
    {
        Baseline lib(f, in.const_view(), out.view(), kBatch, in.image_stride(), out.image_stride(), border, s,
                     extra...);
        lib.run();
    }
    catch (const baseline::Unsupported& e)
    {
        std::printf("  %s r=%d %s: unsupported (%s)\n", Baseline::name(), f.radius(), border_name(border), e.what());
        return -1;
    }

    const std::vector<float> gpu = out.download(s);
    std::vector<float>  ref_out(per_image);
    std::vector<double> mag(per_image);
    const int taps = f.width() * f.width();

    long bad = 0;
    for (int b = 0; b < kBatch; ++b)
    {
        const std::size_t off = static_cast<std::size_t>(b) * per_image;
        ref::correlate(host.data() + off, kW, kH, f.data(), f.radius(), border, ref_out.data(), mag.data());
        bad += ref::compare(gpu.data() + off, ref_out.data(), mag.data(), kW, kH, taps).failing_pixels;
    }
    return bad;
}

void test_cudnn()
{
    const int radii[] = {1, 3, 7, 15};
    for (int r : radii)
    {
        const Filter f = random_filter(r);
        const long   bad = compare_baseline<baseline::CudnnConv>(f, BorderMode::Zero);
        CHECK(bad == 0, "cudnn r=" + std::to_string(r) + " zero: " + std::to_string(bad) + " bad pixels");
    }
    // Any other border must be refused as Unsupported, not computed wrongly.
    CHECK(compare_baseline<baseline::CudnnConv>(random_filter(1), BorderMode::Clamp) == -1,
          "cudnn refuses clamp");
}

void test_npp()
{
    const int radii[] = {1, 3, 7, 15};
    for (int r : radii)
    {
        const Filter f = random_filter(r);
        const long flipped   = compare_baseline<baseline::NppConv>(f, BorderMode::Clamp, true);
        const long unflipped = compare_baseline<baseline::NppConv>(f, BorderMode::Clamp, false);
        CHECK(flipped == 0, "npp r=" + std::to_string(r) + " clamp, flipped kernel: " +
                                std::to_string(flipped) + " bad pixels (want 0)");
        CHECK(unflipped > 0, "npp r=" + std::to_string(r) +
                                 " clamp, UNflipped kernel must disagree (proves NPP convolves)");
    }
    CHECK(compare_baseline<baseline::NppConv>(random_filter(1), BorderMode::Zero, true) == -1,
          "npp refuses zero border");
}

void test_opencv()
{
#ifdef T_CONVL_WITH_OPENCV
    const int radii[] = {1, 3, 7, 15};
    const BorderMode modes[] = {BorderMode::Zero, BorderMode::Clamp, BorderMode::Reflect101};
    for (int r : radii)
    {
        const Filter f = random_filter(r);
        for (BorderMode m : modes)
        {
            const long bad = compare_baseline<baseline::OpencvConv>(f, m);
            if (bad == -1)
                continue; // reported as unsupported; a limit, not an error
            CHECK(bad == 0, "opencv r=" + std::to_string(r) + " " + border_name(m) + ": " +
                                std::to_string(bad) + " bad pixels");
        }
    }
#else
    std::printf("  opencv: not built (define T_CONVL_WITH_OPENCV and link OpenCV's CUDA modules)\n");
#endif
}

bool gpu_available()
{
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess)
    {
        std::printf("no usable GPU: %s\n", cudaGetErrorString(err));
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

} // namespace

int main()
{
    if (!gpu_available())
    {
        std::printf("baselines: GPU checks SKIPPED\n");
        return 77;
    }

    Context ctx;
    g_ctx = &ctx;
    test::run(test_cudnn, "test_cudnn");
    test::run(test_npp, "test_npp");
    test::run(test_opencv, "test_opencv");
    return test::finish("baselines");
}
