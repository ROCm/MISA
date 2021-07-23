#ifndef UTILS_MATH_HPP__
#define UTILS_MATH_HPP__ 1

#include <stdint.h>

template<typename T> static inline T ceil(T val, int factor)
{
    return (val + factor - 1) / factor;
}

template<typename T> static inline T quantize_up(T val, int factor)
{
    return factor * ceil(val, factor);
}

static inline uint32_t rotl(const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

class xoroshiro64
{
private:
    uint32_t s[2];
    float fl;
    double dl;
    int64_t m;

public:
    xoroshiro64(uint32_t seed, int64_t minv = 0, int64_t maxv = 100) {
        s[0] = 0x9E3779BB ^ seed;
        s[1] = 0xBB97379E ^ seed;
        dl = maxv - minv + 1;
        fl = maxv - minv + 1;
        m = minv;
    }

    // random u32
    uint32_t next(void) {
        const uint32_t s0 = s[0];
        uint32_t s1 = s[1];
        const uint32_t result = s0 * 0x9E3779BB;

        s1 ^= s0;
        s[0] = rotl(s0, 26) ^ s1 ^ (s1 << 9); // a, b
        s[1] = rotl(s1, 13); // c

        return result;
    }

    // int from [minv; maxv] interval
    // upper bound is inexact for ranges > 2^23
    int32_t interval_23bit() {
        return m + (int32_t)(f0to1() * fl);
    }

    // int from [minv; maxv] interval
    int32_t interval_32bit() {
        return m + (int32_t)(d0to1_32bit() * dl);
    }

    // int from [minv; maxv] interval
    // upper bound is inexact for ranges > 2^52
    int64_t interval_52bit() {
        return m + (int64_t)(d0to1() * dl);
    }

    // float from 0 to 1
    float f0to1() {
        union {
            float f;
            uint32_t u;
        } fu;

        fu.u = 0x3f800000 | (next() >> 9);
        return fu.f - 1.0f;
    }

    // double from 0 to 1
    // with reduced to 32 bit mantissa precision
    double d0to1_32bit() {
        union {
            double d;
            uint64_t u;
        } du;

        uint64_t v = next();
        du.u = 0x3FF0000000000000L | (v << 20) | (v >> 12);
        return du.d - 1.0;
    }

    // double from 0 to 1
    double d0to1() {
        union {
            double d;
            uint64_t u;
        } du;

        du.u = (0x3FF0000000000000L | next()) ^ (((uint64_t)next()) << 20);
        return du.d - 1.0;
    }
};

#endif