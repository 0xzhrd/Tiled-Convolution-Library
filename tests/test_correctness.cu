/*
Correctness of both tiled kernels (conv_halo.cuh, conv_cache.cuh) against the host reference.

Part 1 (host only): both launchers reject bad requests before anything reaches the GPU.
Part 2 (GPU): known-answer tests (identity, convention, determinism), then a sweep over
radius x border x tile shape x input type x image size x batch, every result compared with
reference::correlate under the per-pixel rounding bound of reference::compare.

Build (one kernel per radius/border/tile/type combination; expect a slow compile):
  nvcc -std=c++14 -I include -I tests \
    -gencode arch=compute_87,code=sm_87 -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 tests/test_correctness.cu -o build/test_correctness
Exit code: 0 = pass, 1 = failure, 77 = host checks passed but no GPU (CTest "skipped").
*/

#include "t_convl/boundary.hpp"
#include "t_convl/convl_cache.cuh"
#include "t_convl/convl_halo.cuh"
#include "t_convl/device_buffer.hpp"
#include "t_convl/filter_bank.cuh"
#include "t_convl/reference.hpp"
#include "check.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using namespace tiled_convolution;
namespace ref = tiled_convolution::reference;

namespace
{

// ---------------------------------------------------------------------------
// Kernel tags: one struct per variant with the same static launch(). Every test below is
// written once against a Kernel template parameter and runs for both variants.
// ---------------------------------------------------------------------------
struct HaloKernel
{
    static const char* name() { return "halo"; }

    template <int R, typename Border, int TX, int TY, typename T>
    static void launch(Image<const T> in, Image<float> out, int slot, cudaStream_t s,
                       int batch = 1, std::ptrdiff_t in_stride = 0, std::ptrdiff_t out_stride = 0)
    {
        conv_halo<R, Border, TX, TY>(in, out, slot, s, batch, in_stride, out_stride);
    }
};

struct CacheKernel
{
    static const char* name() { return "cache"; }

    template <int R, typename Border, int TX, int TY, typename T>
    static void launch(Image<const T> in, Image<float> out, int slot, cudaStream_t s,
                       int batch = 1, std::ptrdiff_t in_stride = 0, std::ptrdiff_t out_stride = 0)
    {
        conv_cache<R, Border, TX, TY>(in, out, slot, s, batch, in_stride, out_stride);
    }
};

// ===========================================================================
// Part 1: launcher validation (no GPU needed: every case throws before launching)
// ===========================================================================
template <typename Kernel>
void launcher_validation()
{
    // Host arrays stand in for device memory: the launcher must reject these requests
    // before it ever launches, so the pointers are never dereferenced.
    static float in_host[64 * 64];
    static float out_host[64 * 64];

    const auto in   = make_view<const float>(in_host, 32, 32);
    const auto out  = make_view(out_host, 32, 32);
    const auto out2 = make_view(out_host, 31, 32);   // wrong width
    const auto tiny = make_view<const float>(in_host, 10, 10);
    const auto tiny_out = make_view(out_host, 10, 10);

    EXPECT_THROW_AS(std::invalid_argument, (Kernel::template launch<1, BorderClamp, 16, 16>(in, out2, 0, nullptr)));
    EXPECT_THROW_AS(std::out_of_range, (Kernel::template launch<1, BorderClamp, 16, 16>(in, out, -1, nullptr)));
    EXPECT_THROW_AS(std::out_of_range, (Kernel::template launch<1, BorderClamp, 16, 16>(in, out, filter_slots, nullptr)));
    EXPECT_THROW_AS(std::invalid_argument, (Kernel::template launch<1, BorderClamp, 16, 16>(in, out, 0, nullptr, 0)));
    EXPECT_THROW_AS(std::invalid_argument, (Kernel::template launch<1, BorderClamp, 16, 16>(in, out, 0, nullptr, 65536)));

    // batch 2 with strides smaller than one image: images would overlap.
    EXPECT_THROW_AS(std::invalid_argument, (Kernel::template launch<1, BorderClamp, 16, 16>(in, out, 0, nullptr, 2, 100, 32 * 32)));
    EXPECT_THROW_AS(std::invalid_argument, (Kernel::template launch<1, BorderClamp, 16, 16>(in, out, 0, nullptr, 2, 32 * 32, 100)));

    // Reflect101 needs width and height > radius: 10x10 with R = 12 is illegal.
    EXPECT_THROW_AS(std::invalid_argument, (Kernel::template launch<12, BorderReflect101, 16, 16>(tiny, tiny_out, 0, nullptr)));
}

void test_launcher_validation()
{
    launcher_validation<HaloKernel>();
    launcher_validation<CacheKernel>();
}

// ===========================================================================
// Part 2: GPU
// ===========================================================================

// Everything the GPU tests share. Created once, after a GPU has been found.
struct Context
{
    Stream       stream;
    FilterBank   bank;
    std::mt19937 rng{20261104u}; // fixed seed: a failure reproduces exactly on the next run
};

constexpr int kSlot = 0;

// Random filter in [-1, 1]: asymmetric, so a flipped (convolution) kernel fails.
Filter random_filter(Context& ctx, int r)
{
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int width = 2 * r + 1;
    std::vector<float> taps(static_cast<std::size_t>(width) * static_cast<std::size_t>(width));
    for (float& t : taps)
        t = dist(ctx.rng);
    return Filter("random_r" + std::to_string(r), r, taps);
}

// Random input of type T: floats in [-1, 1], bytes in [0, 255].
std::vector<float> random_pixels(Context& ctx, std::size_t n, float)
{
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float& p : v)
        p = dist(ctx.rng);
    return v;
}

std::vector<std::uint8_t> random_pixels(Context& ctx, std::size_t n, std::uint8_t)
{
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<std::uint8_t> v(n);
    for (std::uint8_t& p : v)
        p = static_cast<std::uint8_t>(dist(ctx.rng));
    return v;
}

const char* type_name(float) { return "f32"; }
const char* type_name(std::uint8_t) { return "u8"; }

// Where a failing pixel sits: the first clue to which part of the kernel is wrong.
const char* locate(int x, int y, int w, int h, int r, int tx, int ty)
{
    if (x < r || y < r || x >= w - r || y >= h - r)
        return "within R of the image edge -> border policy or halo load";
    if (x % tx == 0 || y % ty == 0 || x % tx == tx - 1 || y % ty == ty - 1)
        return "on a tile edge -> halo load or tile origin";
    return "tile interior -> convention, tap index or accumulation";
}

// One configuration: run the kernel on a pitched (batched) image and compare every image
// in the batch with the reference.
template <typename Kernel, int R, typename Border, int TX, int TY, typename T>
void run_case(Context& ctx, const Filter& filter, int w, int h, int batch)
{
    if (!Border::supports(w, R) || !Border::supports(h, R))
        return; // outside the policy's contract; rejection is tested in Part 1

    const std::size_t per_image = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    const std::vector<T> host_in = random_pixels(ctx, per_image * static_cast<std::size_t>(batch), T{});

    PitchedImage<T>     in(w, h, batch);
    PitchedImage<float> out(w, h, batch);
    in.upload(host_in, ctx.stream.get());
    out.poison(ctx.stream.get()); // any pixel the kernel fails to write stays NaN

    Kernel::template launch<R, Border, TX, TY>(in.const_view(), out.view(), kSlot, ctx.stream.get(),
                                               batch, in.image_stride(), out.image_stride());

    const std::vector<float> gpu = out.download(ctx.stream.get()); // synchronises; kernel errors surface here

    std::vector<float>  ref_out(per_image);
    std::vector<double> mag(per_image);
    const int taps = (2 * R + 1) * (2 * R + 1);

    for (int b = 0; b < batch; ++b)
    {
        const std::size_t offset = static_cast<std::size_t>(b) * per_image;

        ref::correlate(host_in.data() + offset, w, h, filter.data(), R, Border::mode,
                       ref_out.data(), mag.data());
        const ref::CompareResult c =
            ref::compare(gpu.data() + offset, ref_out.data(), mag.data(), w, h, taps);

        char ctx_msg[320];
        std::snprintf(ctx_msg, sizeof ctx_msg,
                      "%s R=%d %s tile=%dx%d %s %dx%d batch %d/%d: %ld bad px, worst ratio %.3g at (%d,%d) %s",
                      Kernel::name(), R, border_name(Border::mode), TX, TY, type_name(T{}), w, h, b, batch,
                      c.failing_pixels, c.worst_ratio, c.worst_x, c.worst_y,
                      c.failing_pixels ? locate(c.worst_x, c.worst_y, w, h, R, TX, TY) : "");
        CHECK(c.failing_pixels == 0, ctx_msg);
    }
}

// Image sizes: odd and unrelated (partial tiles, no width/height symmetry), an exact
// multiple of the tile, a thin strip, and a batched case.
struct Size
{
    int w, h, batch;
};
const Size kSizes[] = {{37, 23, 1}, {16, 16, 1}, {17, 33, 1}, {64, 9, 1}, {100, 61, 3}};

// Every configuration of one kernel variant for one radius.
template <typename Kernel, int R>
void sweep_kernel(Context& ctx, const Filter& filter)
{
    for (const Size& s : kSizes)
    {
        run_case<Kernel, R, BorderZero,       16, 16, float>(ctx, filter, s.w, s.h, s.batch);
        run_case<Kernel, R, BorderClamp,      16, 16, float>(ctx, filter, s.w, s.h, s.batch);
        run_case<Kernel, R, BorderReflect101, 16, 16, float>(ctx, filter, s.w, s.h, s.batch);
        run_case<Kernel, R, BorderReflect101, 32,  8, float>(ctx, filter, s.w, s.h, s.batch);
        run_case<Kernel, R, BorderZero,       32,  8, float>(ctx, filter, s.w, s.h, s.batch);
        run_case<Kernel, R, BorderClamp,      16, 16, std::uint8_t>(ctx, filter, s.w, s.h, s.batch);
    }
}

// One radius: one filter, uploaded once, run through both variants.
template <int R>
void sweep_radius(Context& ctx)
{
    const Filter filter = random_filter(ctx, R);
    ctx.bank.upload(kSlot, filter, ctx.stream.get());
    ctx.bank.require(kSlot, R);

    sweep_kernel<HaloKernel, R>(ctx, filter);
    sweep_kernel<CacheKernel, R>(ctx, filter);
}

// ---------------------------------------------------------------------------
// Known answers: identity must be exact, the convention must be correlation,
// and two identical runs must produce identical bits.
// ---------------------------------------------------------------------------
template <typename Kernel>
void known_answers_for(Context& ctx)
{
    const int w = 37, h = 23;
    std::vector<float> img(static_cast<std::size_t>(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            img[static_cast<std::size_t>(y) * w + x] = static_cast<float>(10 * y + x);

    PitchedImage<float> in(w, h), out(w, h);
    in.upload(img, ctx.stream.get());

    // Identity, radius 1 and 15: output must equal input bit for bit.
    {
        std::vector<float> taps(9, 0.0f);
        taps[4] = 1.0f;
        ctx.bank.upload(kSlot, Filter("identity_r1", 1, taps), ctx.stream.get());
        out.poison(ctx.stream.get());
        Kernel::template launch<1, BorderClamp, 16, 16>(in.const_view(), out.view(), kSlot, ctx.stream.get());
        CHECK(out.download(ctx.stream.get()) == img, std::string(Kernel::name()) + ": identity r=1 exact");
    }
    {
        std::vector<float> taps(31 * 31, 0.0f);
        taps[15 * 31 + 15] = 1.0f;
        ctx.bank.upload(kSlot, Filter("identity_r15", 15, taps), ctx.stream.get());
        out.poison(ctx.stream.get());
        Kernel::template launch<15, BorderZero, 16, 16>(in.const_view(), out.view(), kSlot, ctx.stream.get());
        CHECK(out.download(ctx.stream.get()) == img, std::string(Kernel::name()) + ": identity r=15 exact");
    }

    // Convention: a tap at (+1, 0) must read the RIGHT neighbour (correlation).
    {
        std::vector<float> taps(9, 0.0f);
        taps[1 * 3 + 2] = 1.0f;
        ctx.bank.upload(kSlot, Filter("shift_x", 1, taps), ctx.stream.get());
        out.poison(ctx.stream.get());
        Kernel::template launch<1, BorderClamp, 16, 16>(in.const_view(), out.view(), kSlot, ctx.stream.get());
        const std::vector<float> r = out.download(ctx.stream.get());
        char msg[120];
        std::snprintf(msg, sizeof msg, "%s shift +x: out(5,3)=%g, want 36 (34 means convolution)",
                      Kernel::name(), static_cast<double>(r[3 * w + 5]));
        CHECK(r[3 * w + 5] == 36.0f, msg);
    }

    // Determinism: same input, same filter, two runs -> identical bits.
    {
        Filter f = random_filter(ctx, 7);
        ctx.bank.upload(kSlot, f, ctx.stream.get());
        Kernel::template launch<7, BorderReflect101, 16, 16>(in.const_view(), out.view(), kSlot, ctx.stream.get());
        const std::vector<float> first = out.download(ctx.stream.get());
        out.poison(ctx.stream.get());
        Kernel::template launch<7, BorderReflect101, 16, 16>(in.const_view(), out.view(), kSlot, ctx.stream.get());
        const std::vector<float> second = out.download(ctx.stream.get());
        CHECK(std::memcmp(first.data(), second.data(), first.size() * sizeof(float)) == 0,
              std::string(Kernel::name()) + ": two runs are bit-identical");
    }
}

// The two variants differ only in where halo pixels come from; on identical input they must
// agree to within rounding. (Not bit-identical: the compiler may order FMAs differently.)
void halo_matches_cache(Context& ctx)
{
    const int w = 100, h = 61;
    const Filter f = random_filter(ctx, 9);
    ctx.bank.upload(kSlot, f, ctx.stream.get());

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> img(static_cast<std::size_t>(w) * h);
    for (float& p : img)
        p = dist(ctx.rng);

    PitchedImage<float> in(w, h), a(w, h), b(w, h);
    in.upload(img, ctx.stream.get());
    convl_halo<9, BorderReflect101>(in.const_view(), a.view(), kSlot, ctx.stream.get());
    convl_cache<9, BorderReflect101>(in.const_view(), b.view(), kSlot, ctx.stream.get());
    const std::vector<float> ra = a.download(ctx.stream.get());
    const std::vector<float> rb = b.download(ctx.stream.get());

    std::vector<float>  ref_out(img.size()); // only mag is needed: it scales the allowed difference
    std::vector<double> mag(img.size());
    ref::correlate(img.data(), w, h, f.data(), 9, BorderMode::Reflect101, ref_out.data(), mag.data());
    // Each variant may be up to gamma(n) * mag from the exact answer, so the two may differ from
    // EACH OTHER by up to twice that: pass 2n as the tap count to double the allowance.
    const ref::CompareResult c = ref::compare(ra.data(), rb.data(), mag.data(), w, h, 2 * 19 * 19);
    CHECK(c.failing_pixels == 0, "halo and cache variants agree within twice the rounding bound");
}

void test_known_answers(Context& ctx)
{
    known_answers_for<HaloKernel>(ctx);
    known_answers_for<CacheKernel>(ctx);
    halo_matches_cache(ctx);
}

void test_gpu_sweep(Context& ctx)
{
    sweep_radius<1>(ctx);
    sweep_radius<2>(ctx);
    sweep_radius<3>(ctx);
    sweep_radius<5>(ctx);
    sweep_radius<7>(ctx);
    sweep_radius<8>(ctx);  // largest radius PMPP's block sizing can launch (16x16 tiles)
    sweep_radius<9>(ctx);  // first radius where it cannot: proves the output-sized block
    sweep_radius<15>(ctx); // largest supported radius
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

// run() takes a plain function; these wrappers own the Context for the GPU tests.
Context* g_ctx = nullptr;
void known_answers() { test_known_answers(*g_ctx); }
void gpu_sweep() { test_gpu_sweep(*g_ctx); }

} // namespace

int main()
{
    test::run(test_launcher_validation, "test_launcher_validation");

    if (!gpu_available())
    {
        std::printf("correctness: %d failures in %d host checks; GPU checks SKIPPED\n",
                    test::failures(), test::checks());
        return test::failures() == 0 ? 77 : 1;
    }

    Context ctx;
    g_ctx = &ctx;
    test::run(known_answers, "test_known_answers");
    test::run(gpu_sweep, "test_gpu_sweep");

    return test::finish("correctness");
}
