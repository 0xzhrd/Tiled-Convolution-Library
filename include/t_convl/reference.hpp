/*
host only reference that every GPU kernel is checked against

*/
#pragma once

#include <vector>
#include <cstddef>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "t_convl/boundary.hpp"

namespace tiled_convolution
{
    namespace reference
    {
        inline int filter_width(int rad)  {return(2 * rad + 1);}

        inline int resolve(int coord, int n, BorderMode mode, bool &valid)
        {
            valid = true;
            
            switch(mode)
            {
                case BorderMode::Zero:
                    if(coord < 0 || coord >= n)
                    {
                        valid = false;
                        return(0);
                    }

                case BorderMode::Clamp:
                    if(coord < 0)
                        return(0);
                    if(coord > n - 1)
                    {
                        return(n - 1);
                    }
                    return(coord);
                
                case BorderMode::Reflect101:
                    if(coord < 0)
                    {
                        coord = - coord;
                    }
                    if(coord > (n - 1))
                    {
                        coord = 2 * (n - 1) - coord;
                    }
                    return(coord);
                
            }
            valid = false;
            return(0);
        }

        inline void check_args(int w, int h, int rad, BorderMode mode)
        {
            //image size check
            if(w <= 0 || h <= 0)
                throw std::invalid_argument("reference: non-positive size " + std::to_string(w) + "x" + std::to_string(h));

            if(rad < 0 || rad > max_filter_radius)
                throw std::invalid_argument("reference: radius less that 0 or greater than " + std::to_string(max_filter_radius));

            if(mode == BorderMode::Reflect101 && (w <= rad || h <= rad))
                throw std::invalid_argument("reference: too short in any made");
        }

        template <typename T>
        void correlate(const T *in, int w, int h, const float *filter, int rad, BorderMode mode, float *out, double *abs_sum)
        {
            check_args(w, h, rad, mode);

            if((in || filter || out || abs_sum) == nullptr)
                throw std::invalid_argument("reference: null buffer");

            const int fw = filter_width(rad);
            for(int y = 0; y < h; ++y)
            {
                for(int x = 0; x < w; ++x)
                {
                    double sum = 0.0;
                    double mag = 0.0;

                    for(int dy = -rad; dy <= rad; ++dy)
                    {
                        bool row_ok = false;
                        const int sy = resolve(y + dy, h, mode, row_ok);
                        if(!row_ok)
                            continue;

                        for(int dx = -rad; dx <= rad; ++dx)
                        {
                            bool col_ok = false;
                            const int sx = resolve(x + dx, w, mode, col_ok);
                            if(!col_ok)
                                continue;

                            const double p = static_cast<double>(in[static_cast<std::size_t>(sy) * w + sx]);
                            const double f = static_cast<double>(filter[(dy + rad) * fw + (dy + rad )]);

                            sum += f * p;
                            mag += std::fabs(f) * std::fabs(p);
                        }
                    }
                    const std::size_t i = static_cast<std::size_t>(y) * w + x;
                    out[i] = static_cast<float>(sum);
                    abs_sum[i] = mag;
                }
            }
        }

        inline std::vector<float> flip(const float *filter, int rad)
        {
            const int taps = filter_width(rad) * filter_width(rad);
            std::vector<float> flipped(static_cast<std::size_t>(taps));

            for(int k = 0; k < taps; ++k)
            {
                flipped[static_cast<std::size_t>(taps - 1 - k)] = filter[k];
            }
            return(flipped);
        }

        inline double gamma_fp32(int n)
        {
            const double u = static_cast<double> (std::numeric_limits<float>::epsilon()) / 2.0;
            const double nu = n * u;

            return(nu/(1.0 - nu));
        }
        
        struct CompareResult
        {
            double max_abs_err;
            double worst_ratio;
            int worst_x;
            int worst_y;
            long failing_pixels;
        };
        inline CompareResult compare(const float *gpu, const float *ref, const double *abs_sum, int w, int h, int taps)
        {
            const double g = gamma_fp32(taps + 1);
            const double floor = static_cast<double>(std::numeric_limits<float>::min());

            CompareResult res{0.0, 0.0, -1, -1, 0};

            for(int y = 0; y < h; ++y)
            {
                for(int x = 0; x < w; ++x)
                {
                    const std::size_t i = static_cast<std::size_t>(y) * w + x;
                    const double err = std::fabs(static_cast<double>(gpu[i] - static_cast<double>(ref[i])));
                    const double bound = g * abs_sum[i] + floor;
                    double ratio = err/bound;

                    if(std::isnan(ratio))
                        ratio = std::numeric_limits<double>::infinity();

                    if(err > res.max_abs_err)
                        res.max_abs_err = err;

                    if(ratio > res.worst_ratio)
                    {
                        res.worst_ratio = ratio;
                        res.worst_x = x;
                        res.worst_y = y;
                    }

                    if(ratio > 1.0)
                        ++res.failing_pixels;
                }
            }
            return(res);
        }

    }
}

