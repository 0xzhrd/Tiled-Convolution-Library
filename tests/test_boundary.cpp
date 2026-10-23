// Host-only test for tiled_convolution/boundary.hpp.
// Needs no GPU and no CUDA toolkit:
//   g++ -std=c++14 -Wall -Wextra -Werror -I include tests/test_boundary.cpp -o build/test_boundary
// Also compile under nvcc to exercise the __host__ __device__ branch of the macro:
//   nvcc -std=c++14 -I include -x cu tests/test_boundary.cpp -o build/test_boundary_nvcc
//
// Exit code 0 = all checks passed, 1 = at least one failure. Tools read the exit code.
// Deliberately NOT using assert(): it compiles to nothing under NDEBUG (Release builds).

#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "t_convl/boundary.hpp"

using namespace tiled_convolution;

// ---------------------------------------------------------------------------
// Minimal check machinery: count failures, print only failures, keep going.
// ---------------------------------------------------------------------------
namespace
{
int g_checks   = 0;
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

// Expect the callable NOT to throw.
template <typename F>
void expect_no_throw(F&& f, const char* what, int line)
{
    ++g_checks;
    try
    {
        f();
    }
    catch (const std::exception& e)
    {
        ++g_failures;
        std::printf("FAIL line %d: %s threw unexpectedly: %s\n", line, what, e.what());
    }
    catch (...)
    {
        ++g_failures;
        std::printf("FAIL line %d: %s threw a non-std exception\n", line, what);
    }
}

#define EXPECT_THROW(expr) expect_invalid_argument([&] { (void)(expr); }, #expr, __LINE__)
#define EXPECT_OK(expr)    expect_no_throw([&] { (void)(expr); }, #expr, __LINE__)

// Context buffer for failure messages inside loops.
char g_ctx[160];

// ---------------------------------------------------------------------------
// Naive oracles: written from the definitions, sharing no code with the header.
// ---------------------------------------------------------------------------
int oracle_clamp(int x, int n)
{
    if (x < 0)
        return 0;
    else if (x > n - 1)
        return n - 1;
    else
        return x;
}

// Mirror without repeating the edge. Valid for overshoot <= n - 1 (the contract).
int oracle_reflect101(int x, int n)
{
    if (x < 0)
        x = -x;                 // -1 -> 1, -2 -> 2
    if (x > n - 1)
        x = 2 * (n - 1) - x;    // n -> n - 2
    return x;
}

bool oracle_inside(int x, int n)
{
    return x >= 0 && x < n;
}

// ---------------------------------------------------------------------------
// Test 1: every remap value against the oracles, across many sizes.
// ---------------------------------------------------------------------------
void test_remap_exhaustive()
{
    for (int n = 2; n <= 12; ++n)
    {
        const int r = n - 1; // largest radius Reflect101 supports on this size
        for (int x = -r; x <= (n - 1) + r; ++x)
        {
            const int refl     = BorderReflect101::remap(x, n);
            const int clamp    = BorderClamp::remap(x, n);
            const bool inside  = BorderZero::inside(x, n);

            std::snprintf(g_ctx, sizeof g_ctx, "n=%d x=%d refl=%d(exp %d) clamp=%d(exp %d) inside=%d(exp %d)",
                          n, x, refl, oracle_reflect101(x, n), clamp, oracle_clamp(x, n),
                          inside, oracle_inside(x, n));

            CHECK(refl == oracle_reflect101(x, n), g_ctx);
            CHECK(clamp == oracle_clamp(x, n), g_ctx);
            CHECK(inside == oracle_inside(x, n), g_ctx);

            // Independent property: remapped indices always land inside the image.
            CHECK(refl >= 0 && refl <= n - 1, g_ctx);
            CHECK(clamp >= 0 && clamp <= n - 1, g_ctx);
        }
    }

    // Clamp and Zero promise correctness for ANY overshoot; Reflect101 does not
    // (its limit is enforced by check_border_contract), so it is not tested here.
    const int extremes[] = {-1000000, -1, 1000000, std::numeric_limits<int>::min(),
                            std::numeric_limits<int>::max()};
    for (int x : extremes)
    {
        std::snprintf(g_ctx, sizeof g_ctx, "extreme x=%d n=8", x);
        CHECK(BorderClamp::remap(x, 8) == oracle_clamp(x, 8), g_ctx);
        CHECK(BorderZero::inside(x, 8) == oracle_inside(x, 8), g_ctx);
    }
}

// ---------------------------------------------------------------------------
// Test 2: load() on a real, pitched, NON-square buffer.
// 8 wide x 5 tall, pitch 10: a width/height swap or a pitch bug changes answers.
// ---------------------------------------------------------------------------
constexpr int   kW        = 8;
constexpr int   kH        = 5;
constexpr int   kPitch    = 10;
constexpr float kSentinel = -1000.0f; // value no real pixel can have

float encode(int x, int y) { return static_cast<float>(10 * y + x); }

template <typename Policy, typename View>
void sweep_load(const View& v, const char* policy_name)
{
    const int r = (kW < kH ? kW : kH) - 1; // largest radius Reflect101 allows here (4)

    for (int y = -r; y <= (kH - 1) + r; ++y)
    {
        for (int x = -r; x <= (kW - 1) + r; ++x)
        {
            const float got = Policy::load(v, x, y);

            float expected;
            if (Policy::mode == BorderMode::Zero)
                expected = (oracle_inside(x, kW) && oracle_inside(y, kH)) ? encode(x, y) : 0.0f;
            else if (Policy::mode == BorderMode::Clamp)
                expected = encode(oracle_clamp(x, kW), oracle_clamp(y, kH));
            else
                expected = encode(oracle_reflect101(x, kW), oracle_reflect101(y, kH));

            std::snprintf(g_ctx, sizeof g_ctx, "%s load(x=%d, y=%d): got %g, expected %g",
                          policy_name, x, y, static_cast<double>(got), static_cast<double>(expected));

            // Exact compare is correct: values are small integers stored exactly,
            // and load() copies rather than computes.
            CHECK(got == expected, g_ctx);
            CHECK(got != kSentinel, g_ctx); // read the row padding
        }
    }
}

void test_load_pitched()
{
    float buf[kH * kPitch];

    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kPitch; ++x)
            buf[y * kPitch + x] = (x < kW) ? encode(x, y) : kSentinel;

    // Read-only view (what kernels see) and writable view: both instantiations must work.
    const auto in  = make_view<const float>(buf, kW, kH, kPitch * sizeof(float));
    const auto out = make_view<float>(buf, kW, kH, kPitch * sizeof(float));

    CHECK(in.width == kW, "pitched view width");
    CHECK(in.height == kH, "pitched view height");
    CHECK(in.pitch == kPitch, "pitch stored in ELEMENTS, not bytes");
    CHECK(in.data == buf, "pitched view data pointer");

    // row() must step by pitch, not width.
    CHECK(in.row(3) == buf + 3 * kPitch, "row(3) steps by pitch");

    sweep_load<BorderZero>(in, "zero");
    sweep_load<BorderClamp>(in, "clamp");
    sweep_load<BorderReflect101>(in, "reflect101");

    sweep_load<BorderZero>(out, "zero(non-const)");
    sweep_load<BorderClamp>(out, "clamp(non-const)");
    sweep_load<BorderReflect101>(out, "reflect101(non-const)");
}

// ---------------------------------------------------------------------------
// Test 3: factory validation. Each check in make_view gets its own case,
// plus the boundary case that must NOT throw.
// ---------------------------------------------------------------------------
void test_factory()
{
    float buf[kH * kPitch] = {};
    float* const null_ptr  = nullptr;

    EXPECT_THROW(make_view(null_ptr, kW, kH));            // null data
    EXPECT_THROW(make_view(buf, 0, kH));                  // zero width
    EXPECT_THROW(make_view(buf, kW, 0));                  // zero height
    EXPECT_THROW(make_view(buf, -8, kH, 64));             // negative width
    EXPECT_THROW(make_view(buf, kW, kH, 30));             // 30 bytes: not a multiple of sizeof(float)
    EXPECT_THROW(make_view(buf, kW, kH, 28));             // 7 elements < width 8

    const std::size_t huge_pitch =
        (static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1) * sizeof(float);
    EXPECT_THROW(make_view(buf, kW, kH, huge_pitch));     // pitch does not fit in int

    EXPECT_OK(make_view(buf, kW, kH, kW * sizeof(float))); // pitch == width: tightest legal

    // Contiguous overload: pitch must equal width, in elements.
    const auto c = make_view(buf, kW, kH);
    CHECK(c.pitch == kW, "contiguous overload: pitch == width");
    CHECK(c.width == kW && c.height == kH, "contiguous overload: dimensions");
}

// ---------------------------------------------------------------------------
// Test 4: check_border_contract at its limits, both dimensions.
// ---------------------------------------------------------------------------
void test_contract()
{
    EXPECT_OK(check_border_contract<BorderReflect101>(16, 16, 15));   // n = r + 1: tightest legal
    EXPECT_THROW(check_border_contract<BorderReflect101>(15, 16, 15)); // width too small
    EXPECT_THROW(check_border_contract<BorderReflect101>(16, 15, 15)); // height too small

    EXPECT_OK(check_border_contract<BorderClamp>(2, 2, 15));           // no size limit
    EXPECT_OK(check_border_contract<BorderZero>(2, 2, 15));            // no size limit

    EXPECT_THROW(check_border_contract<BorderZero>(16, 16, -1));       // negative radius
    EXPECT_THROW(check_border_contract<BorderClamp>(16, 16, -1));
    EXPECT_THROW(check_border_contract<BorderReflect101>(16, 16, -1));
}

} // namespace

int main()
{
    test_remap_exhaustive();
    test_load_pitched();
    test_factory();
    test_contract();

    std::printf("boundary: %d failures in %d checks\n", g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}