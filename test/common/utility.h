#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <random>
#include <assert.h>
#include <cmath>

#define RAND_MAX_FLOAT  1.0
#define RAND_MIN_FLOAT  -1.0
#ifndef ABS
#define ABS(x) ((x)>0?(x):-1*(x))
#endif
#ifndef MAX
#define MAX(a,b)    ( (a)>(b)?(a):(b) )
#endif

#define RMS_THRESHOLD 1e-6


static bool is_power_of_2(int x)
{
  return x > 0 && !(x & (x-1));
}

template<typename T>
static bool valid_float(T p)
{
    return ! (std::isnan(p) || std::isinf(p));
}

template<typename T>
static void rand_vec(T *  seq, size_t len){
    static std::random_device rd;   // seed
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<T> dist(RAND_MIN_FLOAT, RAND_MAX_FLOAT);
    for(size_t i=0;i<len;i++)
        seq[i] =  dist(mt);
}

template<typename T>
static int valid_vector(const T* lhs, const T* rhs, size_t len, T delta = (T)3e-6){
    int err_cnt = 0;
    for(size_t i = 0;i < len; i++){
        T d = lhs[i]- rhs[i];
        d = ABS(d);
        if(!(valid_float(lhs[i]) && valid_float(rhs[i]))){
            std::cout<<" invalid float at "<<i<<", lhs:"<<lhs[i]<<", rhs:"<<rhs[i]<<std::endl;
            err_cnt++;
            continue;
        }
        if(d > delta){
            std::cout<<" diff at "<<i<<", lhs:"<<lhs[i]<<", rhs:"<<rhs[i]<<", delta:"<<d<<std::endl;
            err_cnt++;
        }
    }
    return err_cnt;
}
template<typename T>
static int valid_vector_nrms(const T* pred, const T* ref, size_t len, double tolerance = (double)RMS_THRESHOLD)
{
    double v, max, nrms;
    v = 0;
    max = std::numeric_limits<double>::min();
    for(size_t i=0;i<len;i++){
        if(!(valid_float(ref[i]) && valid_float(pred[i]))){
            return -1;
        }
        double d = ref[i]-pred[i];
        double m2 = MAX(ABS(ref[i]),ABS(pred[i]));
        v += d*d;
        max = MAX(max,m2);
    }
    nrms = sqrt(v)/(sqrt(len)*max);
    return (nrms<tolerance)?0:1;
}

template<typename T>
static void dump_vector(const T * vec, size_t len){
    for(size_t i=0;i<len;i++) std::cout<<vec[i]<<(i==(len-1)?"":", ");
    std::cout<<std::endl;
}

template<typename T>
static void dump_vector_as_py_array(const T * vec, size_t len){
    assert(is_power_of_2(len));
    std::cout<<"[";
    for(size_t i=0;i<len;i++)
        if(i % 2 == 0)
            std::cout<<vec[i];
        else{
            T v = vec[i];
            std::cout<<(v >= (T)0 ? "+":"")<<v<<"j";
            if(i != (len - 1))
                std::cout<<", ";
        }
    std::cout<<"]"<<std::endl;
}

#endif
