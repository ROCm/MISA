#ifndef FFT_H
#define FFT_H

#include <assert.h>
#include <tuple>
#include <vector>
#include <math.h>
#include <functional>


#define PRE_PAD_DATA
#define FFTCONV_USE_CONJ // this is a good mode that all omega use the same function, unified_omega_func_f32
#define FFTCONV_USE_CONJ_NO_ROTATE // this mode, all kernel padding shape is same. we restore output in c2r part
//#define FFTCONV_USE_CONJ_A  // same as FFTCONV_USE_CONJ, but notice, time reverse is fft shift
#define MERGE_2D_NYQUEST_FREQ

#if defined(FFTCONV_USE_CONJ) && defined(FFTCONV_USE_CONJ_A)
#   error "can't both conj and conj_a mode"
#endif

#ifndef C_PI
#define C_PI  3.14159265358979323846
#endif
#ifndef C_2PI
#define C_2PI 6.28318530717958647692
#endif


#define LD_C(vec,idx,r,i) do{r=vec[2*(idx)];i=vec[2*(idx)+1];}while(0)
#define ST_C(vec,idx,r,i) do{vec[2*(idx)]=r;vec[2*(idx)+1]=i;}while(0)

// A=ar+ai*i, B=br+bi*i, omega=omr+omi*i
// A'= A+omega*B = ar+ai*i+(omr+omi*i)*(br+bi*i) = ar+omr*br-omi*bi + (ai+omi*br+omr*bi)*i
// B'= A-omega*B = ar+ai*i-(omr+omi*i)*(br+bi*i) = ar-omr*br+omi*bi + (ai-omr*bi-omi*br)*i
#define BTFL_C(ar,ai,br,bi,omr,omi,tr,ti) do{\
    tr=br*omr-bi*omi;ti=br*omi+bi*omr; \
    br=ar; bi=ai;\
    ar=ar+tr;ai=ai+ti;\
    br=br-tr;bi=bi-ti; } while(0)

// A=ar+ai*i, B=br+bi*i, omega=omr+omi*i
// A'= A+conj(omega)*B = ar+ai*i+(omr-omi*i)*(br+bi*i) = ar+omr*br+omi*bi + (ai-omi*br+omr*bi)*i
// B'= A-conj(omega)*B = ar+ai*i-(omr-omi*i)*(br+bi*i) = ar-omr*br-omi*bi + (ai-omr*bi+omi*br)*i
#define IBTFL_C(ar,ai,br,bi,omr,omi,tr,ti) do{\
    tr=br*omr+bi*omi;ti=-br*omi+bi*omr; \
    br=ar; bi=ai;\
    ar=ar+tr;ai=ai+ti;\
    br=br-tr;bi=bi-ti; } while(0)

int bit_reverse_nbits(int v, int nbits){
    int r = 0; int d = nbits-1;
    for(int i=0;i<nbits;i++)
    {   if(v & (1<<i)) r |= 1<<d;  d--; }
    return r;
}

// below function produce  https://oeis.org/A030109
void bit_reverse_permute(size_t radix2_num, std::vector<size_t> &arr)
{
    arr.resize(pow(2,radix2_num));
    arr[0] = 0;
    for(size_t k=0;k<radix2_num;k++){
       size_t last_k_len = pow(2, k);
       size_t last_k;
       for(last_k = 0; last_k < last_k_len; last_k++){
           arr[last_k] = 2*arr[last_k];
           arr[last_k_len+last_k] = arr[last_k]+1;
       }
    }
}

template<typename T>
void bit_reverse_radix2_c(T *vec,size_t c_length){
    assert( ( (c_length & (c_length - 1)) == 0 ) && "must be radix of 2");
    std::vector<size_t> r_idx;
    bit_reverse_permute(log2(c_length), r_idx);
    for(size_t i=0;i<c_length;i++){
        size_t ir = r_idx[i];
        if(i<ir)
            { std::swap(vec[2*i], vec[2*ir]); std::swap(vec[2*i+1], vec[2*ir+1]); }
    }
}

// inplace transpose, seq has c_length complex value, 2*c_length value
template<typename T>
void _fft_cooley_tukey_r(T * seq, size_t c_length, bool is_inverse_fft, bool need_final_reverse = true){
    if(c_length == 1) return;
    assert( ( (c_length & (c_length - 1)) == 0 ) && "current only length power of 2");

    std::function<std::tuple<T,T>(size_t,size_t)> omega_func;
#if defined(FFTCONV_USE_CONJ) || defined(FFTCONV_USE_CONJ_A)
    omega_func = [](size_t total_n, size_t k){
            T theta = -1*C_2PI*k / total_n;
            return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta)); };
#else
    if(is_inverse_fft){
        omega_func = [](size_t total_n, size_t k){
            T theta = C_2PI*k / total_n;
            return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta)); };
    }else{
        omega_func = [](size_t total_n, size_t k){
            T theta = -1*C_2PI*k / total_n;
            return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta)); };
    }
#endif

    for(size_t itr = 2; itr<=c_length; itr<<=1){
        size_t stride = c_length/itr;
        size_t groups = itr/2;
        size_t group_len = stride*2;

        std::vector<std::tuple<T,T>> omega_list;   // pre-compute omega, and index to it later
        omega_list.resize(itr/2);
        for(size_t i = 0; i < itr/2 ; i ++){
            omega_list[i] = omega_func( itr, i);
        }
        for(size_t g=0;g<groups;g++){
            size_t k = bit_reverse_nbits(g, log2(groups));  
            T omr, omi;
            std::tie(omr,omi) = omega_list[k];
            for(size_t s=0;s<stride;s++){
                T ar,ai,br,bi,tr,ti;
                LD_C(seq,g*group_len+s,ar,ai);
                LD_C(seq,g*group_len+s+stride,br,bi);
#if defined(FFTCONV_USE_CONJ) || defined(FFTCONV_USE_CONJ_A)
                if(is_inverse_fft)
                    IBTFL_C(ar,ai,br,bi,omr,omi,tr,ti);
                else
                    BTFL_C(ar,ai,br,bi,omr,omi,tr,ti);
#else
                BTFL_C(ar,ai,br,bi,omr,omi,tr,ti);
#endif
                ST_C(seq,g*group_len+s,ar,ai);
                ST_C(seq,g*group_len+s+stride,br,bi);
            }
        }
    }
    if(need_final_reverse)
        bit_reverse_radix2_c(seq, c_length);
    if(is_inverse_fft){
        for(size_t i=0;i<c_length;i++){
            seq[2*i] = seq[2*i]/c_length;
            seq[2*i+1] = seq[2*i+1]/c_length;
        }
    }
}
template<typename T>
void fft_cooley_tukey_r(T * seq, size_t c_length, bool need_final_reverse = true){
    _fft_cooley_tukey_r(seq, c_length, false, need_final_reverse);
}
template<typename T>
void ifft_cooley_tukey_r(T * seq, size_t c_length, bool need_final_reverse = true){
    _fft_cooley_tukey_r(seq, c_length, true, need_final_reverse);
}


#endif
