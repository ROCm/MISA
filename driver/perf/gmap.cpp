/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

#include "perf.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <initializer_list>
#include <cstdint>
#include <tuple>
#include <algorithm>
#include <iterator>

using index_t = uint64_t;

typedef struct {
    index_t tid;
    index_t data_byte;      // 1, 2, 4
    index_t vector;         // x1, x2, x4...
    index_t offset;      // start offset of this request, in byte
    bool valid;          // if this request is valid, aka, within tensor range
} req_t;

typedef struct {
    index_t bid;
    index_t block_size;
    index_t req_idx;            // requst index, a counter for the request
    std::vector<req_t> req; // request of each thread
} block_req_t;

void serialize_block_req(const block_req_t * block_req, FILE* fp, std::vector<bool> * record)
{
    fprintf(fp, "[b:%d,r:%d] ", block_req->bid, block_req->req_idx);
    assert(block_req->block_size == block_req->req.size());
    for(int i=0; i<block_req->req.size(); i++){
        const auto & thread_req = block_req->req[i];
        assert(thread_req.tid == i);
        fprintf(fp, "t%d:", i);
        for(int v=0; v<thread_req.vector; v++){
            index_t offset = thread_req.offset + v * thread_req.data_byte;
            index_t ipixel = offset / thread_req.data_byte;
            if(record){
                if((*record)[ipixel]){
                    printf("have already visited this pixel:%zu, offset:%zu\n",ipixel,offset);
                    assert(0);
                }
                (*record)[ipixel] = true;
            }
            fprintf(fp, "0x%zx", offset);
            if(v != (thread_req.vector - 1))
                fprintf(fp, ",");
        }
        fprintf(fp, "(%s) ", thread_req.valid ? "y":"n");
    }
    fprintf(fp, "\n");
    fflush(fp);
}



class linear_tensor_t{
public:
    linear_tensor_t(std::initializer_list<index_t> _dims):dims(_dims){}

    // get nd indices from a linear index
    std::vector<index_t> get(index_t linear_index){
        std::vector<index_t> nd_index(dims.size(), (index_t)0);
        index_t len = 1;
        auto  rind_itr = std::rbegin(nd_index);
        for(auto  rdim_itr = dims.rbegin();
                    rdim_itr != dims.rend();
                    rdim_itr++, rind_itr++){
            *rind_itr = (linear_index / len) % *rdim_itr;
            len *= *rdim_itr;
        }

        return nd_index;
    }
    // get offset from nd indices
    index_t offset(std::initializer_list<index_t> indices)
    {
        assert(indices.size() == dims.size());
        index_t stride = 1;
        index_t len = 0;
        auto rind_itr = std::rbegin(indices);
        for(auto  rdim_itr = dims.rbegin();
                    rdim_itr != dims.rend();
                    rdim_itr++, rind_itr++){
            len += *rind_itr * stride;
            stride *= *rdim_itr;
        }
        return len;
    }

    // nd range check
    bool range_check(std::initializer_list<index_t> indices)
    {
        assert(indices.size() == dims.size());
        bool valid = true;
        auto rind_itr = std::rbegin(indices);
        for(auto  rdim_itr = dims.rbegin();
                    rdim_itr != dims.rend();
                    rdim_itr++, rind_itr++){
            valid &= *rind_itr < *rdim_itr;
        }
        return valid;
    }
private:
    std::vector<index_t> dims;
};

static inline index_t gmap_conv_out_size(index_t in_size, index_t pad, index_t dilation,
                                   index_t ksize, index_t stride) {
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}

#define GMAP_DIR "gmap/"

std::tuple<std::string, std::string, std::string>
gmap_get_dump_file_name(const std::string base_dir, const igemm_gtc_tunable_t * tunable)
{
    std::string kernel_name = igemm_gtc_encode_kernel_name(tunable);
    return std::make_tuple(base_dir + "/" + std::string("gmap_inp_") + kernel_name +std::string(".dump"),
                           base_dir + "/" + std::string("gmap_wei_") + kernel_name +std::string(".dump"),
                           base_dir + "/" + std::string("gmap_out_") + kernel_name +std::string(".dump"));
}

void gmap_dump_fwd_nhwc(const args_t *conv_args, const igemm_gtc_tunable_t * tunable, FILE *fp_inp, FILE *fp_wei, FILE *fp_out)
{
    index_t hi = conv_args->get_int("in_h");
    index_t wi = conv_args->get_int("in_w");
    index_t n = conv_args->get_int("batchsize");
    index_t k = conv_args->get_int("out_channels");
    index_t c = conv_args->get_int("in_channels");

    index_t stride_h = conv_args->get_int("conv_stride_h");
    index_t stride_w = conv_args->get_int("conv_stride_w");
    index_t dilation_h = conv_args->get_int("dilation_h");
    index_t dilation_w = conv_args->get_int("dilation_w");
    index_t pad_h = conv_args->get_int("pad_h");
    index_t pad_w = conv_args->get_int("pad_w");
    index_t y = conv_args->get_int("fil_h");
    index_t x = conv_args->get_int("fil_w");
    index_t ho = gmap_conv_out_size(hi, pad_h, dilation_h, y, stride_h);
    index_t wo = gmap_conv_out_size(wi, pad_w, dilation_w, x, stride_w);
    index_t group = conv_args->get_int("group_count");

    std::string precision = tunable->precision;
    index_t data_byte = utility_string_to_data_byte(tunable->precision);

    index_t gemm_m_per_block = tunable->gemm_m_per_block;
    index_t gemm_n_per_block = tunable->gemm_n_per_block;
    index_t gemm_k_per_block = tunable->gemm_k_per_block;
    index_t gemm_m = ((n * ho * wo + gemm_m_per_block - 1) / gemm_m_per_block) * gemm_m_per_block;
    index_t gemm_n = k / group;
    index_t gemm_k = (c / group) * y * x;

    index_t ta_e    = tunable->tensor_a_thread_lengths[0];
    index_t ta_c    = tunable->tensor_a_thread_lengths[1];
    index_t ta_nb0  = tunable->tensor_a_thread_lengths[2];
    index_t ta_nb1  = tunable->tensor_a_thread_lengths[3];

    index_t tb_e    = tunable->tensor_b_thread_lengths[0];
    index_t tb_c    = tunable->tensor_b_thread_lengths[1];
    index_t tb_k0   = tunable->tensor_b_thread_lengths[2];
    index_t tb_k1   = tunable->tensor_b_thread_lengths[3];

    index_t ca_e    = tunable->tensor_a_cluster_lengths[0];
    index_t ca_c    = tunable->tensor_a_cluster_lengths[1];
    index_t ca_nb0  = tunable->tensor_a_cluster_lengths[2];
    index_t ca_nb1  = tunable->tensor_a_cluster_lengths[3];

    index_t cb_e    = tunable->tensor_b_cluster_lengths[0];
    index_t cb_c    = tunable->tensor_b_cluster_lengths[1];
    index_t cb_k0   = tunable->tensor_b_cluster_lengths[2];
    index_t cb_k1   = tunable->tensor_b_cluster_lengths[3];

    printf("ta%dx%dx%dx%d_%dx%dx%dx%d, tb%dx%dx%dx%d_%dx%dx%dx%d\n",
                ta_e, ta_c, ta_nb0, ta_nb1, ca_e, ca_c, ca_nb0, ca_nb1,
                tb_e, tb_c, tb_k0 , tb_k1 , cb_e, cb_c, cb_k0 , cb_k1 );

    index_t block_size = ca_e * ca_c * ca_nb0 * ca_nb1;
    assert(block_size == (cb_e * cb_c * cb_k0 * cb_k1));
    assert((gemm_m % gemm_m_per_block == 0) && (gemm_n % gemm_n_per_block == 0));
    index_t grid_size = group * (gemm_m / gemm_m_per_block) * (gemm_n / gemm_n_per_block);
    linear_tensor_t block_mapping({group, (gemm_m / gemm_m_per_block), (gemm_n / gemm_n_per_block)});
    linear_tensor_t gemm_m_transform({n, ho, wo});
    linear_tensor_t gemm_k_transform({y, x, c / group});

    linear_tensor_t tensor_inp({n, hi, wi, c/group});
    linear_tensor_t tensor_wei({group, k/group, y, x, c/group});
    linear_tensor_t tensor_out({n, ho, wo, k/group});
    std::vector<bool> record_inp(n*hi*wi*(c/group), false);
    std::vector<bool> record_wei(group*(k/group)*y*x*(c/group), false);
    std::vector<bool> record_out(n*ho*wo*(k/group), false);

    index_t ta_nb_per_thread = ta_nb0 != 1 ? ta_nb0 : ta_nb1;
    index_t tb_nk_per_thread = tb_k0 != 1 ? tb_k0 : tb_k1;
    index_t ta_vector_c = utility_gcd(ta_c, 4 * (4 / data_byte));
    index_t ta_nc_per_thread = ta_c / ta_vector_c;
    index_t tb_vector_c = utility_gcd(tb_c, 4 * (4 / data_byte));
    index_t tb_nc_per_thread = tb_c / tb_vector_c;
    index_t ta_nb_thread_stride = tunable->tensor_a_pass_through ? ca_nb0 * ca_nb1 : (
                                    ta_nb0 != 1 ? ca_nb1 * ta_nb1 :1);
    
    std::vector<index_t> ta_block_req_idx(grid_size, 0);
    std::vector<index_t> tb_block_req_idx(grid_size, 0);
    auto cur_block = [&](index_t bid, index_t cur_group, index_t cur_gemm_m, index_t cur_gemm_n, index_t cur_gemm_k){
        std::vector<block_req_t> inp_block_req;
        inp_block_req.resize(ta_nb_per_thread * ta_nc_per_thread);

        // inp
        for(index_t t_inb = 0; t_inb < ta_nb_per_thread; t_inb++){
            for(index_t t_ic = 0; t_ic < ta_nc_per_thread; t_ic++){
                index_t i_req = t_inb * ta_nc_per_thread + t_ic;
                block_req_t & b_req = inp_block_req[i_req];
                b_req.block_size = block_size;
                b_req.bid = bid;
                b_req.req_idx = ta_block_req_idx[bid];

                for(index_t tid = 0; tid < block_size; tid++){
                    index_t in_inb, in_ic;
                    index_t in_in, in_iho, in_iwo, in_ihi, in_iwi;
                    if(tunable->tensor_a_pass_through){

                    }else{
                        in_ic   = (tid % ca_c) * ta_c;
                        in_inb  = (tid / ca_c) * ta_nb1;
                    }
                    index_t cur_in_inb = cur_gemm_m + in_inb + t_inb * ta_nb_thread_stride;

                    auto in_gemm_m_trans = gemm_m_transform.get(cur_in_inb);
                    auto in_gemm_k_trans = gemm_k_transform.get(cur_gemm_k);

                    index_t cur_in_iy = in_gemm_k_trans[0];
                    index_t cur_in_ix = in_gemm_k_trans[1];
                    index_t cur_in_ic = in_gemm_k_trans[2] + cur_gemm_n + in_ic + t_ic * ta_vector_c;

                    index_t cur_in_in = in_gemm_m_trans[0];
                    index_t cur_in_iho = in_gemm_m_trans[1];
                    index_t cur_in_iwo = in_gemm_m_trans[2];

                    // ihi = iho * s_stride_h + iy * s_dilation_h - s_pad_h
                    // iwi = iwo * s_stride_w + ix * s_dilation_w - s_pad_w
                    index_t cur_in_ihi = cur_in_iho * stride_h + cur_in_iy * dilation_h - pad_h;
                    index_t cur_in_iwi = cur_in_iwo * stride_w + cur_in_ix * dilation_w - pad_w;

                    auto cur_in_idx = {cur_in_in, cur_in_ihi, cur_in_iwi, cur_in_ic};
                    bool cur_in_valid = tensor_inp.range_check(cur_in_idx);
                    index_t cur_in_offset = tensor_inp.offset(cur_in_idx) * data_byte;
                    b_req.req.emplace_back(req_t({tid, data_byte, ta_vector_c, cur_in_offset, cur_in_valid}));
                }
                ta_block_req_idx[bid]++;
            }
        }
        for(auto itr_ibr = inp_block_req.begin(); itr_ibr != inp_block_req.end(); itr_ibr++)
            serialize_block_req(&(*itr_ibr), fp_inp, &record_inp);
    };

    for(index_t bid = 0; bid < grid_size; bid++){
        auto cur_block_position = block_mapping.get(bid);   // position of this block in ndim space
        auto cur_group  = cur_block_position[0];
        auto cur_gemm_m = cur_block_position[1] * gemm_m_per_block;
        auto cur_gemm_n = cur_block_position[2] * gemm_n_per_block;
        for(index_t cur_gemm_k = 0; cur_gemm_k < gemm_k; cur_gemm_k += gemm_k_per_block){
            cur_block(bid, cur_group, cur_gemm_m, cur_gemm_n, cur_gemm_k);
        }
    }

}

// global memory access pattern
void gmap_dump(const args_t *conv_args, const igemm_gtc_tunable_t * tunable)
{
    int err = mkdir(GMAP_DIR, 0775);
    if(err != 0){
        if(errno == EEXIST){
            printf("WARNING: directory %s already exist. will dump into it anyway.\n", GMAP_DIR);
        }else{
            printf("[%d]%s: fail to creat directory\n", errno, strerror(errno));
            return ;
        }
    }

    std::string gmap_file_inp;
    std::string gmap_file_wei;
    std::string gmap_file_out;
    std::tie(gmap_file_inp, gmap_file_wei, gmap_file_out) = gmap_get_dump_file_name(GMAP_DIR, tunable);

    FILE * fp_inp = fopen(gmap_file_inp.c_str(), "w");
    if(!fp_inp){
        printf("[%d]%s: fail to open file %s\n", errno, strerror(errno), gmap_file_inp.c_str());
        return ;
    }

    FILE * fp_wei = fopen(gmap_file_wei.c_str(), "w");
    if(!fp_wei){
        printf("[%d]%s: fail to open file %s\n", errno, strerror(errno), gmap_file_wei.c_str());
        return ;
    }

    FILE * fp_out = fopen(gmap_file_out.c_str(), "w");
    if(!fp_out){
        printf("[%d]%s: fail to open file %s\n", errno, strerror(errno), gmap_file_out.c_str());
        return ;
    }


    std::string tensor_layout = tunable->tensor_layout;
    std::string precision = tunable->precision;
    std::string direction = tunable->direction;

    if(direction == "fwd"){
        if(tensor_layout == "nchw"){

        }else if(tensor_layout == "nhwc"){
            gmap_dump_fwd_nhwc(conv_args, tunable, fp_inp, fp_wei, fp_out);
        }else{
            assert(0);
        }
    }

    fclose(fp_inp);
    fclose(fp_wei);
    fclose(fp_out);
}
