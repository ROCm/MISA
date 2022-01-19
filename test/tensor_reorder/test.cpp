#include <iostream>
#include <string>
# define assert(expr)		(__ASSERT_VOID_CAST (0))
using namespace std;
template<int size>
struct sequence{
    sequence(){
    printf("size=%d", size);
    }
};
typedef struct {
    uint32_t magic;
    uint8_t shift;
} magic_div_u32_t;

static inline magic_div_u32_t magic_div_u32_gen(uint32_t d) {
    //assert(d >= 1 && d <= INT32_MAX);
    uint8_t shift;
    for (shift = 0; shift < 32; shift++)
        if ((1U << shift) >= d)
            break;

    uint64_t one = 1;
    uint64_t magic = ((one << 32) * ((one << shift) - d)) / d + 1;
    //assert(magic <= 0xffffffffUL);

    magic_div_u32_t result;
    result.magic = magic;
    result.shift = shift;
    return result;
}
inline uint32_t magic_div_u32(const uint32_t& numer,
                                         const uint32_t& magic,
                                         const uint32_t& shift)
{
    uint32_t tmp = ( ((unsigned long)numer * (unsigned long)magic ) >> 32) & 0x00000000FFFFFFFF;

    return (tmp + numer) >> shift;
}
template <typename T, typename dst_order>
inline  void general_4d_reorder(T* dst,
                                  T* src,
                                  uint32_t dim_0,
                                  uint32_t dim_1,
                                  uint32_t dim_2,
                                  uint32_t dim_3,
                                  uint32_t dim_stride,
                                  uint32_t dim_total,
                                  uint32_t magic_h,
                                  uint32_t shift_h,
                                  uint32_t magic_w,
                                  uint32_t shift_w)
{
    /*
     * assume src is 0, 1, 2, 3, dst is Sequence(0, 1, 2, 3)
     */
     constexpr auto dorder = dst_order{};
     uint32_t src_index =0, dst_index=0;
     uint32_t dim_src[4] = {dim_1 * dim_2 *dim_3, 
                                    dim_2 *dim_3, 
                                           dim_3, 
                                               1};
     uint32_t dim_dst[4] = {dim_src[dorder.At(0)], 
                            dim_src[dorder.At(1)],
                            dim_src[dorder.At(2)],
                            dim_src[dorder.At(3)]};

     uint32_t i_src[4] = {0, 0, 0, 0};
     uint32_t i_dst[4] = {i_src[dorder.At(0)], 
                          i_src[dorder.At(1)],
                          i_src[dorder.At(2)],
                          i_src[dorder.At(3)]};
    
    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        for (uint32_t k = 0; k < 16; k++)
        {
                                //unroll k         block          thread
            src_index = k*dim_total*256 + dim_id * 256 + threadIdx.x;
            i_src[0] = (src_index)/dim_src[0];
            i_src[1] = (src_index -  i_src[0] * dim_src[0])/dim_src[1];
            i_src[2] = (src_index -  i_src[0] * dim_src[0] -  i_src[1] * dim_src[1])/dim_src[2];
            i_src[3] = (src_index -  i_src[0] * dim_src[0] -  i_src[1] * dim_src[1] -  i_src[2] * dim_src[2])/dim_src[3];

            i_dst[0] = i_src[dorder.At(0)];
            i_dst[1] = i_src[dorder.At(1)];
            i_dst[2] = i_src[dorder.At(2)];
            i_dst[3] = i_src[dorder.At(3)];

            dst_index = i_dst[0] * dim_dst[0] + i_dst[1] * dim_dst[1] + i_dst[2] * dim_dst[2] + i_dst[3] * dim_dst[3]; 
            dst[dst_index] = src[src_index];
        }
    }
}
int main(int argc, char ** argv)
{   
    //int w_dim = 128;
    //int h_dim = 128;
    //int dim_id = 10;
    //magic_div_u32_t h =magic_div_u32_gen(15);
    //magic_div_u32_t w =magic_div_u32_gen(6);
    //uint32_t dim_ih_tmp = magic_div_u32(dim_id, w.magic, w.shift);
    //uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
    //uint32_t dim_in     = magic_div_u32(dim_ih_tmp, h.magic, h.shift);
    //uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;
    //printf("magic: %d, shift: %x\n", w.magic, w.shift);
    //printf("dim_ih_tmp: %d, dim_iw: %d, dim_in: %d, dim_ih %d\n", dim_ih_tmp, dim_iw, dim_in, dim_ih);
    
    int x = 5;
    int& y =x;
    std::cout<<x<<','<<y<<std::endl;
    y=7;
    std::cout<<x<<','<<y<<std::endl;
    int order[4] = {2, 1, 0, 3};
    int src[4] = {0, 0, 0, 0};
    int* dst[4] = {&(src[order[0]]), &(src[order[1]]), &(src[order[2]]), &(src[order[3]])};
    for (size_t i = 0; i < 10; i++)
    {
        src[0]+=1; 
        src[1]+=2; 
        src[2]+=3; 
        src[3]+=4;
        printf("src:%d, %d, %d, %d\n", src[0], src[1], src[2], src[3]);
        printf("dst:%d, %d, %d, %d\n", *(dst[0]), *(dst[1]), *(dst[2]), *(dst[3]));
        //printf("dst:%d, %d, %d, %d\n", dst[0], dst[1], dst[2], dst[3]);
    }
    
    return 0;
}