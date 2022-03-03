/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef _TENSOR_VALIDATION_H
#define _TENSOR_VALIDATION_H

#include <stdint.h>
#include <stddef.h>

#ifndef ABS
#define ABS(x) ((x) > 0 ? (x) : -1 * (x))
#endif

template<typename T>
bool valid_float(T p)
{
    return !(std::isnan(p) || std::isinf(p));
}

template<>
bool valid_float<int8_t>(int8_t p)
{
    // there is no meaning to valid integer number
    return true;
}

template<>
bool valid_float<int4x2_t>(int4x2_t p)
{
    // there is no meaning to valid integer number
    return true;
}

template<typename T>
bool valid_vector(const float *ref, const T *pred, size_t n,
                                double nrms = 1.5e-6) {
    double s0 = 0.0;
    double s1 = 0.0;
    int igemm_per_pixel_check = env_get_int("PER_PIXEL_CHECK", 0);
    int igemm_per_pixel_check_print = env_get_int("PER_PIXEL_CHECK_PRINT", 1);
    int print_every_pixel = env_get_int("PRINT_EVERY_PIXEL", 0);
    int print_nrms = env_get_int("PRINT_NRMS", 0);
    int igemm_valid_float = env_get_int("VALID_FLOAT", 1);
    int dump_pred_dword = env_get_int("DUMP_PRED", 0);
    size_t pp_err = 0;

    if(dump_pred_dword){
        // dump as dword, weather the type of pred
        size_t total_safe_size = n / ( sizeof(float) / sizeof(T) );
        for(size_t i=0; i<total_safe_size;i++ ){
            printf("[%zu] ref:%lf, pred:0x%08x\n", i, ref[i], ((uint32_t*)pred)[i]);
        }
    }
#if USE_MIOPEN_NRMS
    double square_difference = .0;
    double mag1 = .0;
    double mag2 = .0;
    for (size_t i = 0; i < n; ++i) {
        if(igemm_valid_float)
            if(!(valid_float<float>(ref[i]) && valid_float<T>(pred[i]))){
                printf(" invalid float at %zu, ref:%f, pred:%f\n", i, ref[i], pred[i]);
                return false;
            }
        

        double ri = (double)ref[i];
        double pi = (double)pred[i];
        double d = ri - pi;

        if(igemm_per_pixel_check){
            double delta = ABS(ABS(ri - pi) / ri);      // TODO: this is just a reference compare
            if(print_every_pixel)
                printf("[%zu] ref:%lf, pred:%lf(0x%08x) [%s]\n", i, ri, pi, *(uint32_t*)(&pred[i]), delta > 3e-5? "N":"Y");
            if (delta > 3e-5) {
                if(igemm_per_pixel_check_print){
                    if (pp_err < 100)
                        printf("diff at %zu, ref:%lf, pred:%lf(0x%08x), d:%lf\n", i, ri,
                            pi, *(uint32_t*)(&pred[i]), delta);
                }
                pp_err++;
            }
        }

        square_difference += d * d;
        if(ABS(mag1) < ABS(ri)) mag1 = ri;
        if(ABS(mag2) < ABS(pi)) mag2 = pi;
    }
    double mag = std::max({std::fabs(mag1), std::fabs(mag2), std::numeric_limits<double>::min()});
    double computed_nrms = std::sqrt(square_difference) / (std::sqrt(n) * mag);
    if(print_nrms)
        printf("\nnrms:%lf, mag1:%lf, mag2:%lf, expected_nrms is %1f\n",computed_nrms,mag1,mag2,nrms);
    return (computed_nrms < nrms)
#ifdef PER_PIXEL_CHECK
           && (pp_err == 0)
#endif
        ;
#else
    for (size_t i = 0; i < n; ++i) {
        if(igemm_valid_float)
            if(!(valid_float<float>(ref[i]) && valid_float<T>(pred[i]))){
                printf(" invalid float at %zu, ref:%f, pred:%f\n", i, ref[i], pred[i]);
                return false;
            }
        double ri = (double)ref[i];
        double pi = (double)pred[i];
        double d = ri - pi;
        double dd = d * d;
        double rr = 2.0 * ri * ri;
        s0 += dd;
        s1 += rr;
        if(igemm_per_pixel_check){
            double delta = ABS(ABS(ri - pi) / ri);
            printf("[%zu] ref:%lf, pred:%lf(0x%08x) [%s]\n", i, ri, pi, *(uint32_t*)(&pred[i]), delta > 3e-5? "N":"Y");
            if (delta > 3e-5) {
                if(igemm_per_pixel_check_print){
                    if (pp_err < 100)
                        printf("diff at %zu, ref:%lf, pred:%lf(0x%08x), d:%lf\n", i, ri,
                            pi, *(uint32_t*)(&pred[i]), delta);
                }
                pp_err++;
            }

        }
    }
    if(print_nrms)
        printf("\nnrms:%lf, s0:%lf, s1:%lf, expected_nrms is %1f\n",sqrt(s0/s1),s0,s1,nrms);
    return (sqrt(s0 / s1) < nrms)
#ifdef PER_PIXEL_CHECK
           && (pp_err == 0)
#endif
        ;
#endif
}

template<>
bool valid_vector<int8_t>(const float *ref, const int8_t *pred, size_t n,
                                double nrms) {
    // int8 valid, we prefer a per pixel match
    int igemm_per_pixel_check = env_get_int("PER_PIXEL_CHECK", 0);
    int igemm_per_pixel_check_print = env_get_int("PER_PIXEL_CHECK_PRINT", 1);
    int dump_pred_dword = env_get_int("DUMP_PRED", 0);
    size_t pp_err = 0;

    if(dump_pred_dword){
        // dump as dword, weather the type of pred
        size_t total_safe_size = n / ( sizeof(float) / sizeof(int8_t) );
        for(size_t i=0; i<total_safe_size;i++ ){
            printf("[%zu] ref:%lf, pred:0x%08x\n", i, ref[i], ((uint32_t*)pred)[i]);
        }
    }

    for (size_t i = 0; i < n; ++i) {
        if(!(valid_float<float>(ref[i]) ) ){
            printf(" invalid float at %4zu, ref:%f\n", i, ref[i]);
            return false;
        }
        int8_t pi = pred[i];
        int32_t ri = static_cast<int32_t>(ref[i]);
        int8_t ri_clamp;
        memcpy(&ri_clamp, &ri, 1);

        if(igemm_per_pixel_check){
            printf("[%zu] ref:%d(%d), pred:%d(0x%08x) [%s]\n", i, ri, ri_clamp, pi,
                        *(uint32_t*)(&pred[i]), pi != ri_clamp ? "N":"Y");
        }

        if(pi != ri_clamp){
            pp_err++;
        }
    }
    return pp_err == 0;
}

template<>
bool valid_vector<int4x2_t>(const float *ref, const int4x2_t *pred, size_t n, double nrms) 
{
    // int8 valid, we prefer a per pixel match
    int igemm_per_pixel_check = env_get_int("PER_PIXEL_CHECK", 0);
    int igemm_per_pixel_check_print = env_get_int("PER_PIXEL_CHECK_PRINT", 1);
    int dump_pred_dword = env_get_int("DUMP_PRED", 0);
    size_t pp_err = 0;

    if(dump_pred_dword){
        // dump as dword, weather the type of pred
        size_t total_safe_size = n / ( sizeof(float) / sizeof(int8_t) );
        for(size_t i=0; i<total_safe_size;i++ ){
            printf("[%zu] ref:%lf, pred:0x%08x\n", i, ref[i], ((uint32_t*)pred)[i]);
        }
    }

    int8_t *tmp_pred = (int8_t *)pred;

    for (size_t i = 0; i < (n / 2); ++i) {
        if(!(valid_float<float>(ref[2 * i]) && valid_float<float>(ref[2 * i + 1]))){
            printf(" invalid float at %4zu, ref:%f, %f\n", 2 * i, ref[2 * i], ref[2 * i + 1]);
            return false;
        }
        
        int8_t pi = tmp_pred[i];
        int32_t ri_lo = static_cast<int32_t>(ref[2 * i]);
        int32_t ri_hi = static_cast<int32_t>(ref[2 * i + 1]);
        int8_t ri_clamp;
        ri_lo = ri_lo & 0xf;
        ri_hi = ri_hi & 0xf;
        ri_clamp = (ri_hi << 4) + ri_lo;

        if(pi != ri_clamp)
        {
            if(igemm_per_pixel_check)
            {
                if (pp_err < 100)
                {
                    printf("[%zu] ref:%d, %d(0x%x), tmp_pred:%d(0x%x) [%s]\n", 2 * i, ri_lo, ri_hi, ri_clamp, pi,
                        *(uint32_t*)(&tmp_pred[i]), pi != ri_clamp ? "N":"Y");
                }
            }
        }

        if(pi != ri_clamp){
            pp_err++;
        }
    }
    return pp_err == 0;
}

double get_nrms(std::string direction, driverDataType_t driver_data_type){
    auto basic_tolerance = [=]() -> double{
        if (driver_data_type == driverFloat){
#ifdef USE_XDNN
            return 5e-5;
#else
            return 1.5e-6;
#endif
        }
        else if (driver_data_type == driverHalf){
#ifdef USE_XDNN
            return 5*8.2e-3;
#else
            return 8.2e-3;
#endif
        }
    };
    double nrms = basic_tolerance();
    if (direction == "bwd"){
        // nrms *= 10;
    }
    // wrw has a high tolerance
    if (direction == "wrw"){
        nrms *= 2;
        if(driver_data_type == driverFloat){
            nrms = 0.01;
        }
        else if(driver_data_type == driverHalf){
            nrms *= 5;
        }
    }
    return nrms;
}

#endif