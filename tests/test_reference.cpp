/*
Host-only test of reference.hpp.
*/

#include "t_convl/reference.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace tiled_convolution;
namespace ref = tiled_convolution::reference;

namespace // everything inside this namespace is private to this file
{
    char g_ctx[200];
    int  g_checks   = 0;
    int  g_failures = 0;

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
        catch(const std::invalid_argument&)
        {
            return; // pass
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
        std::printf("FAIL line %d: %s did not throw\n", line, what);
    }

    #define EXPECT_THROW(expr) expect_invalid_argument([&] { (void)(expr); }, #expr, __LINE__)

    const BorderMode kModes[] = {BorderMode::Zero, BorderMode::Clamp, BorderMode::Reflect101};

    // Odd, unrelated dimensions: not multiples of 16, and a width/height swap shows up.
    constexpr int kW = 37;
    constexpr int kH = 23;

    // Every pixel encodes its own coordinates: value 10*y + x.
    std::vector<float> coord_image()
    {
        const std::size_t size = static_cast<std::size_t>(kW) * kH;
        std::vector<float> result(size);
        for (int y = 0; y < kH; ++y)
        {
            for (int x = 0; x < kW; ++x)
            { // row major
                result[static_cast<std::size_t>(y) * kW + static_cast<std::size_t>(x)] =
                    static_cast<float>(10 * y + x);
            }
        }
        return result;
    }

    float at(const std::vector<float>& img, int x, int y)
    {
        return img[static_cast<std::size_t>(y) * kW + static_cast<std::size_t>(x)];
    }

    // Radius-r filter that is zero except for a 1 at offset (dx, dy).
    // Uses the layout formula independently of the reference.
    std::vector<float> delta_filter(int r, int dx, int dy)
    {
        const int fw = ref::filter_width(r);
        std::vector<float> result(static_cast<std::size_t>(fw) * static_cast<std::size_t>(fw), 0.0f);
        result[static_cast<std::size_t>(dy + r) * static_cast<std::size_t>(fw) +
            static_cast<std::size_t>(dx + r)] = 1.0f;
        return result;
    }

    void test_identity()
    {
        const std::vector<float> in = coord_image();
        std::vector<float>  out(in.size());
        std::vector<double> mag(in.size());

        for (int r = 1; r <= 11; ++r) // 11: largest radius Reflect101 allows on a 23-row image
        {
            const std::vector<float> f = delta_filter(r, 0, 0);
            for (BorderMode m : kModes)
            {
                ref::correlate(in.data(), kW, kH, f.data(), r, m, out.data(), mag.data());

                int wrong = 0;
                for (std::size_t i = 0; i < in.size(); ++i)
                    wrong += (out[i] != in[i]);

                std::snprintf(g_ctx, sizeof g_ctx, "identity r=%d mode=%s: %d pixels differ",
                            r, border_name(m), wrong);
                CHECK(wrong == 0, g_ctx);
            }
        }
    }

    void test_shift()
    {
        const std::vector<float> in = coord_image();
        std::vector<float>  out(in.size());
        std::vector<double> mag(in.size());

        // Horizontal: tap at (+1, 0) -> out(x,y) == in(x+1, y).
        {
            const std::vector<float> f = delta_filter(1, 1, 0);
            ref::correlate(in.data(), kW, kH, f.data(), 1, BorderMode::Clamp, out.data(), mag.data());

            std::snprintf(g_ctx, sizeof g_ctx, "shift +x: out(5,3)=%g, want 36 (34 means convolution)",
                        static_cast<double>(at(out, 5, 3)));
            CHECK(at(out, 5, 3) == 36.0f, g_ctx);

            int wrong = 0;
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW - 1; ++x) // last column's neighbour depends on the border
                    wrong += (at(out, x, y) != at(in, x + 1, y));
            std::snprintf(g_ctx, sizeof g_ctx, "shift +x interior: %d pixels wrong", wrong);
            CHECK(wrong == 0, g_ctx);
        }

        // Vertical: tap at (0, +1) -> out(x,y) == in(x, y+1). Catches a dx/dy transpose.
        {
            const std::vector<float> f = delta_filter(1, 0, 1);
            ref::correlate(in.data(), kW, kH, f.data(), 1, BorderMode::Clamp, out.data(), mag.data());
            std::snprintf(g_ctx, sizeof g_ctx, "shift +y: out(5,3)=%g, want 45",
                        static_cast<double>(at(out, 5, 3)));
            CHECK(at(out, 5, 3) == 45.0f, g_ctx);
        }

        // Off-centre tap at a larger radius: (-3, +2), r = 4 -> out(x,y) == in(x-3, y+2).
        {
            const std::vector<float> f = delta_filter(4, -3, 2);
            ref::correlate(in.data(), kW, kH, f.data(), 4, BorderMode::Zero, out.data(), mag.data());
            std::snprintf(g_ctx, sizeof g_ctx, "shift (-3,+2): out(10,5)=%g, want %g",
                        static_cast<double>(at(out, 10, 5)), static_cast<double>(at(in, 7, 7)));
            CHECK(at(out, 10, 5) == at(in, 7, 7), g_ctx);
        }
    }

    void test_box_borders()
    {
        const std::vector<float> in(static_cast<std::size_t>(kW) * kH, 7.0f);
        std::vector<float>  out(in.size());
        std::vector<double> mag(in.size());
        const std::vector<float> box(9, 1.0f / 9.0f);

        // Clamp and Reflect101 read only real pixels (all 7): output is 7 everywhere.
        for (BorderMode m : {BorderMode::Clamp, BorderMode::Reflect101})
        {
            ref::correlate(in.data(), kW, kH, box.data(), 1, m, out.data(), mag.data());
            double worst = 0.0;
            for (float v : out)
                worst = std::fmax(worst, std::fabs(static_cast<double>(v) - 7.0));
            std::snprintf(g_ctx, sizeof g_ctx, "box %s: max |out - 7| = %g", border_name(m), worst);
            CHECK(worst < 1e-5, g_ctx);
        }

        // Zero: a corner sees 4 real pixels of 9, an edge 6, the interior 9.
        ref::correlate(in.data(), kW, kH, box.data(), 1, BorderMode::Zero, out.data(), mag.data());
        const double corner = at(out, 0, 0);
        const double edge   = at(out, 5, 0);
        const double inner  = at(out, 5, 5);
        std::snprintf(g_ctx, sizeof g_ctx,
                    "box zero: corner %g (want %g) edge %g (want %g) inner %g (want 7)",
                    corner, 7.0 * 4 / 9, edge, 7.0 * 6 / 9, inner);
        CHECK(std::fabs(corner - 7.0 * 4 / 9) < 1e-5, g_ctx);
        CHECK(std::fabs(edge - 7.0 * 6 / 9) < 1e-5, g_ctx);
        CHECK(std::fabs(inner - 7.0) < 1e-5, g_ctx);
        CHECK(std::fabs(at(out, kW - 1, kH - 1) - 7.0 * 4 / 9) < 1e-5, "box zero: opposite corner");
    }

    void test_uint8()
    {
        std::vector<std::uint8_t> in(static_cast<std::size_t>(kW) * kH, 200);
        std::vector<float>  out(in.size());
        std::vector<double> mag(in.size());
        const std::vector<float> f = delta_filter(2, 0, 0);

        ref::correlate(in.data(), kW, kH, f.data(), 2, BorderMode::Reflect101, out.data(), mag.data());
        CHECK(out[0] == 200.0f && out[in.size() - 1] == 200.0f, "uint8 identity returns 200");
    }

    void test_flip()
    {
        const std::vector<float> sobel = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        const std::vector<float> f1    = ref::flip(sobel.data(), 1);
        const std::vector<float> f2    = ref::flip(f1.data(), 1);

        bool neg  = true;
        bool back = true;
        for (std::size_t k = 0; k < sobel.size(); ++k)
        {
            neg  = neg && (f1[k] == -sobel[k]);
            back = back && (f2[k] == sobel[k]);
        }
        CHECK(neg, "flip(sobel_x) == -sobel_x");
        CHECK(back, "flip(flip(f)) == f");

        const std::vector<float> gauss = {1, 2, 1, 2, 4, 2, 1, 2, 1};
        CHECK(ref::flip(gauss.data(), 1) == gauss, "symmetric filter unchanged by flip");

        const std::vector<float> in = coord_image();
        std::vector<float>  out(in.size());
        std::vector<double> mag(in.size());
        const std::vector<float> shift   = delta_filter(1, 1, 0);
        const std::vector<float> flipped = ref::flip(shift.data(), 1);
        ref::correlate(in.data(), kW, kH, flipped.data(), 1, BorderMode::Clamp, out.data(), mag.data());
        CHECK(at(out, 5, 3) == 34.0f, "correlate(flip(shift +x)) reads the left neighbour");
    }

    void test_compare()
    {
        const std::vector<float> in    = coord_image();
        const std::vector<float> sobel = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        std::vector<float>  refout(in.size());
        std::vector<double> mag(in.size());
        ref::correlate(in.data(), kW, kH, sobel.data(), 1, BorderMode::Reflect101, refout.data(), mag.data());

        ref::CompareResult c = ref::compare(refout.data(), refout.data(), mag.data(), kW, kH, 9);
        CHECK(c.worst_ratio == 0.0 && c.failing_pixels == 0, "self-compare is exact");

        // One pixel, perturbed by 10x its own bound, must be the only failure, at (11, 7).
        std::vector<float> gpu = refout;
        const std::size_t  i   = static_cast<std::size_t>(7) * kW + 11;
        gpu[i] += static_cast<float>(10.0 * ref::gamma_fp32(10) * mag[i]);
        c = ref::compare(gpu.data(), refout.data(), mag.data(), kW, kH, 9);
        std::snprintf(g_ctx, sizeof g_ctx, "perturbed: %ld failing at (%d,%d), want 1 at (11,7)",
                    c.failing_pixels, c.worst_x, c.worst_y);
        CHECK(c.failing_pixels == 1 && c.worst_x == 11 && c.worst_y == 7, g_ctx);

        // A perturbation at 0.1x the bound passes.
        gpu    = refout;
        gpu[i] = refout[i] + static_cast<float>(0.1 * ref::gamma_fp32(10) * mag[i]);
        c      = ref::compare(gpu.data(), refout.data(), mag.data(), kW, kH, 9);
        CHECK(c.failing_pixels == 0, "perturbation at 0.1x bound passes");

        // NaN must count as a failure.
        gpu    = refout;
        gpu[3] = std::numeric_limits<float>::quiet_NaN();
        c      = ref::compare(gpu.data(), refout.data(), mag.data(), kW, kH, 9);
        CHECK(c.failing_pixels == 1, "NaN counts as a failure");
    }

    void test_preconditions()
    {
        const std::vector<float> in(100, 1.0f);
        std::vector<float>  out(100);
        std::vector<double> mag(100);
        const std::vector<float> f  = delta_filter(1, 0, 0);
        const std::vector<float> f5 = delta_filter(5, 0, 0);

        EXPECT_THROW(ref::correlate(in.data(), 0, 10, f.data(), 1, BorderMode::Zero, out.data(), mag.data()));
        EXPECT_THROW(ref::correlate(in.data(), 10, 10, f.data(), -1, BorderMode::Zero, out.data(), mag.data()));
        EXPECT_THROW(ref::correlate(in.data(), 10, 10, f.data(), 16, BorderMode::Zero, out.data(), mag.data()));
        EXPECT_THROW(ref::correlate(in.data(), 5, 20, f5.data(), 5, BorderMode::Reflect101, out.data(), mag.data()));
        EXPECT_THROW(ref::correlate(in.data(), 10, 10, f.data(), 1, BorderMode::Zero,
                                    static_cast<float*>(nullptr), mag.data()));
        EXPECT_THROW(ref::correlate(static_cast<const float*>(nullptr), 10, 10, f.data(), 1,
                                    BorderMode::Zero, out.data(), mag.data()));
    }
}

int main()
{
    test_identity();
    test_shift();
    test_box_borders();
    test_uint8();
    test_flip();
    test_compare();
    test_preconditions();

    std::printf("reference: %d failures in %d checks\n", g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}