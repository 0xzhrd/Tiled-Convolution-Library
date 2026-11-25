/*
Test of convl2d.cuh: the runtime-to-compile-time dispatch, the variant selector, and the
checks convl2d makes before dispatching. Kernel correctness itself is test_correctness.cu's
job; here a convl2d call must produce EXACTLY the bits the matching direct launcher produces,
because it must have dispatched to that same kernel.

Build:
  nvcc -std=c++14 -I include -I tests \
    -gencode arch=compute_87,code=sm_87 -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 tests/test_convl2d.cu -o build/test_convl2d
Exit code: 0 = pass, 1 = failure, 77 = host checks passed but no GPU.
*/

#include "t_convl/convl2d.cuh"
#include "t_convl/device_buffer.hpp"

#include "check.hpp"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using namespace tiled_convolution;

namespace
{

// ===========================================================================
// Part 1: host only
// ===========================================================================
void test_selector()
{
    const SelectorConfig def;
    CHECK(select_variant(0, def) == Variant::Cache, "r=0 -> cache");
    CHECK(select_variant(def.cache_max_radius, def) == Variant::Cache, "r=threshold -> cache");
    CHECK(select_variant(def.cache_max_radius + 1, def) == Variant::Halo, "r=threshold+1 -> halo");
    CHECK(select_variant(15, def) == Variant::Halo, "r=15 -> halo");

    SelectorConfig custom;
    custom.cache_max_radius = 9;
    CHECK(select_variant(9, custom) == Variant::Cache && select_variant(10, custom) == Variant::Halo,
          "custom threshold 9");

    CHECK(std::string(variant_name(Variant::Halo)) == "halo" &&
              std::string(variant_name(Variant::Cache)) == "cache" &&
              std::string(variant_name(Variant::Auto)) == "auto",
          "variant names");
}

void test_rejections_without_gpu()
{
    static float in_host[32 * 32];
    static float out_host[32 * 32];
    const auto in  = make_view<const float>(in_host, 32, 32);
    const auto out = make_view(out_host, 32, 32);

    FilterBank bank; // no CUDA calls until upload()

    // Radius outside [0, 15] is rejected before the bank is consulted.
    EXPECT_THROW_AS(std::invalid_argument, convl2d(bank, 0, -1, in, out, nullptr));
    EXPECT_THROW_AS(std::invalid_argument, convl2d(bank, 0, max_filter_radius + 1, in, out, nullptr));

    // Legal radius, but nothing uploaded: require() refuses, nothing launches.
    EXPECT_THROW_AS(std::logic_error, convl2d(bank, 0, 3, in, out, nullptr));

    // Bad slot.
    EXPECT_THROW_AS(std::out_of_range, convl2d(bank, filter_slots, 3, in, out, nullptr));
}

// ===========================================================================
// Part 2: GPU
// ===========================================================================
struct Context
{
    Stream       stream;
    FilterBank   bank;
    std::mt19937 rng{7u};
};

constexpr int kSlot = 1;

Filter random_filter(Context& ctx, int r)
{
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int w = 2 * r + 1;
    std::vector<float> taps(static_cast<std::size_t>(w) * static_cast<std::size_t>(w));
    for (float& t : taps)
        t = dist(ctx.rng);
    return Filter("rand_r" + std::to_string(r), r, taps);
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// For one compile-time radius and border: convl2d with each variant choice must match the
// direct launcher bit for bit, and must report the variant it ran.
template <int R, typename Border>
void check_dispatch(Context& ctx, const PitchedImage<float>& in, PitchedImage<float>& out)
{
    const cudaStream_t s = ctx.stream.get();
    const int batch      = in.batch();

    ConvOptions opt;
    opt.border           = Border::mode;
    opt.batch            = batch;
    opt.in_image_stride  = in.image_stride();
    opt.out_image_stride = out.image_stride();

    out.poison(s);
    convl_halo<R, Border>(in.const_view(), out.view(), kSlot, s, batch, in.image_stride(), out.image_stride());
    const std::vector<float> want_halo = out.download(s);

    out.poison(s);
    convl_cache<R, Border>(in.const_view(), out.view(), kSlot, s, batch, in.image_stride(), out.image_stride());
    const std::vector<float> want_cache = out.download(s);

    char msg[160];

    const Variant forced[] = {Variant::Halo, Variant::Cache, Variant::Auto};
    for (Variant v : forced)
    {
        opt.variant = v;
        out.poison(s);
        const Variant ran = convl2d(ctx.bank, kSlot, R, in.const_view(), out.view(), s, opt);
        const std::vector<float> got = out.download(s);

        const Variant expected = (v == Variant::Auto) ? select_variant(R, opt.selector) : v;
        const std::vector<float>& want = (expected == Variant::Halo) ? want_halo : want_cache;

        std::snprintf(msg, sizeof msg, "R=%d %s requested=%s: ran=%s (want %s)", R,
                      border_name(Border::mode), variant_name(v), variant_name(ran), variant_name(expected));
        CHECK(ran == expected, msg);

        std::snprintf(msg, sizeof msg, "R=%d %s requested=%s: output matches the direct launcher bit for bit",
                      R, border_name(Border::mode), variant_name(v));
        CHECK(same_bits(got, want), msg);
    }
}

template <int R>
void check_radius(Context& ctx, const PitchedImage<float>& in, PitchedImage<float>& out)
{
    const Filter f = random_filter(ctx, R);
    ctx.bank.upload(kSlot, f, ctx.stream.get());

    check_dispatch<R, BorderZero>(ctx, in, out);
    check_dispatch<R, BorderClamp>(ctx, in, out);
    check_dispatch<R, BorderReflect101>(ctx, in, out);
}

Context* g_ctx = nullptr;

void test_dispatch()
{
    Context& ctx = *g_ctx;
    const int w = 53, h = 37, batch = 2;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> img(static_cast<std::size_t>(w) * h * batch);
    for (float& p : img)
        p = dist(ctx.rng);

    PitchedImage<float> in(w, h, batch), out(w, h, batch);
    in.upload(img, ctx.stream.get());

    // Both sides of the default threshold (3), a middle radius, PMPP's limit, and the maximum.
    check_radius<0>(ctx, in, out);
    check_radius<3>(ctx, in, out);
    check_radius<4>(ctx, in, out);
    check_radius<9>(ctx, in, out);
    check_radius<15>(ctx, in, out);
}

void test_require_mismatch()
{
    Context& ctx = *g_ctx;
    PitchedImage<float> in(40, 40), out(40, 40);

    ctx.bank.upload(kSlot, random_filter(ctx, 3), ctx.stream.get());
    // Slot holds radius 3; asking for radius 5 must be refused before anything launches.
    EXPECT_THROW_AS(std::logic_error, convl2d(ctx.bank, kSlot, 5, in.const_view(), out.view(), ctx.stream.get()));
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
    test::run(test_selector, "test_selector");
    test::run(test_rejections_without_gpu, "test_rejections_without_gpu");

    if (!gpu_available())
    {
        std::printf("convl2d: %d failures in %d host checks; GPU checks SKIPPED\n",
                    test::failures(), test::checks());
        return test::failures() == 0 ? 77 : 1;
    }

    Context ctx;
    g_ctx = &ctx;
    test::run(test_dispatch, "test_dispatch");
    test::run(test_require_mismatch, "test_require_mismatch");
    return test::finish("convl2d");
}
