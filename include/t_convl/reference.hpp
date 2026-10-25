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

            const int fw = filter_width(r);
            for(int y = 0; y < h; ++y)
            {
                for(int x = 0; x < w; ++x)
                {
                    double sum = 0.0;
                    double mag = 0.0;

                    for(int dy = -r; dy <= r; ++dy)
                    {
                        bool row_ok = false;
                        const int sy = resolve(y + dy, h, mode, row_ok);
                        if(!row_ok)
                            continue;

                        for(int dx = -r; dx <= r; ++dx)
                        {
                            bool col_ok = false;
                            const int sx = resolve(x + dx, w, mode, col_ok);
                            if(!col_ok)
                                continue;

                            const double p = static_cast<double>(in[static_cast<std::size_t>(sy) * w + sx]);

                            
                        }
                    }
                }
            }
        }
    }
}

