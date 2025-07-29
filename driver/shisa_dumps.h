#pragma once

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <float.h>
#include <functional>
#include <cstdint>
#include <stdlib.h>
#include <array>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "common.h"


struct gridinfo_t {
    std::array<uint32_t, 3> gsize;
    std::array<uint32_t, 3> wsize;
};

#define MAX_KARG_DUMP_BYTES 1024
enum class kargtype_t : int {
    unknown                    = 0,
    igemm_fwd_gtc_karg_t       = 1,
    igemm_fwd_gtc_nhwc_karg_t  = 2,
    igemm_fwd_gtc_nchwc_karg_t = 3,
    igemm_bwd_gtc_karg_t       = 4,
    igemm_bwd_gtc_nhwc_karg_t  = 5,
    igemm_wrw_gtc_karg_t       = 6
};

enum class convdir_t : int { FWD = 1, BWD = 2, WRW = 3 };

enum class misadatatype_t : int {
    UNKNOWN = 0,
    FP32 = 1,
    FP16 = 2,
    BF16 = 3,
    INT8 = 4,
    INT4 = 5
};

struct dispatchinfo_t {
    size_t karg_size;
    gridinfo_t gi;
    kargtype_t ktype;
    uint32_t _reserved;
    char karg_dump[MAX_KARG_DUMP_BYTES];
};
static_assert(sizeof(dispatchinfo_t) == MAX_KARG_DUMP_BYTES + 40);

struct convparams_t {
    int hi;
    int wi;
    int n;
    int k; // this is indeed k_per_group
    int c; // this is indeed c_per_group
    int ho;
    int wo;
    int stride_h;
    int stride_w;
    int ddilation_h;
    int ddilation_w;
    int fdilation_h;
    int fdilation_w;
    int pad_h;
    int pad_w;
    int y;
    int x;
    int group;
    convdir_t dir;
    misadatatype_t dtype;
};

#define DUMPFILE_VERSION 4
struct dumpheader_t {

    static dumpheader_t make_header(const args_t *arg, gridinfo_t &postlog,
                                    size_t workspace_size, size_t dispatch_cnt,
                                    int cur_gks, int cast_length, bool use_prolog,
                                    bool use_postlog, convdir_t dir,
                                    misadatatype_t misa_type) {

        int hi = arg->get_int("in_h");
        int wi = arg->get_int("in_w");
        int n = arg->get_int("batchsize");
        int k = arg->get_int("out_channels");
        int c = arg->get_int("in_channels");

        int stride_h = arg->get_int("conv_stride_h");
        int stride_w = arg->get_int("conv_stride_w");
        int dilation_h = arg->get_int("dilation_h");
        int dilation_w = arg->get_int("dilation_w");
        int pad_h = arg->get_int("pad_h");
        int pad_w = arg->get_int("pad_w");
        int y = arg->get_int("fil_h");
        int x = arg->get_int("fil_w");
        int ho = conv_out_size(hi, pad_h, dilation_h, y, stride_h);
        int wo = conv_out_size(wi, pad_w, dilation_w, x, stride_w);
        int group = arg->get_int("group_count");

        convparams_t conv_params{
            hi,       wi,       n, k,     c,          ho,         wo,
            stride_h, stride_w, 1, 1,     dilation_h, dilation_w, pad_h,
            pad_w,    y,        x, group, dir,        misa_type};

        dumpheader_t new_header{
            DUMPFILE_VERSION, dispatch_cnt, workspace_size,
            postlog,          conv_params,  cur_gks,
            use_prolog,       use_postlog,  cast_length};

        return new_header;
    }

    uint64_t version = DUMPFILE_VERSION;
    size_t   n_dispatches;
    size_t   workspace_size;
    gridinfo_t   gi_postlog;
    convparams_t conv;
    int gks;
    int use_prolog;
    int use_postlog;
    int cast_total_length;
};
static_assert(sizeof(dumpheader_t) == 36 * 4);

misadatatype_t dtype(const std::string &s);
void dump_shader_args(std::string dump_dir, const dumpheader_t &header, const std::vector<dispatchinfo_t> &data, std::string kernel_name);

class DumpWriter_t {

  public:
    DumpWriter_t(std::string kernel_name)
        : dump_path(env_get_str("IGEMM_DUMPDIR_ALL", const_cast<char*>(""))),
          kernel_name(kernel_name), is_active(dump_path.size() != 0) {}

    template <typename T>
    void dump_kernels(std::vector<T> &kernels,
                      std::function<dumpheader_t(int, int)> head_builder,
                      int gks, kargtype_t ktype) const {
        if (!is_active)
            return;

        dumpheader_t dumpheader = head_builder(kernels.size(), gks);

        std::vector<dispatchinfo_t> kernel_data;

        for (auto &x : kernels) {
            kernel_data.push_back(x);
            kernel_data.back().ktype = ktype;
        }

        dump_to_file(dumpheader, kernel_data);
    }

  private:
    void dump_to_file(const dumpheader_t &const_header,
                      const std::vector<dispatchinfo_t> &dispatches) const {

        dumpheader_t header = const_header;
        header.version = DUMPFILE_VERSION;

        if (header.n_dispatches != dispatches.size()) {
            std::cout << "ERROR: header.n_dispatches != dispatches.size()"
                      << std::endl;
            assert(0);
        }

        std::string dump_file = dump_path + kernel_name + ".gks" +
                                std::to_string(header.gks) + ".dump";

        std::ofstream fs(dump_file, std::ios::out | std::ios::binary);
        fs.write(reinterpret_cast<const char *>(&header), sizeof(dumpheader_t));

        for (auto &di : dispatches) {
            fs.write(reinterpret_cast<const char *>(&di), sizeof(di));
        }
        fs.write(kernel_name.c_str(), kernel_name.size());
        fs.close();
    }

    std::string dump_path;
    std::string kernel_name;
    bool is_active = false;
};

// return_code : -1, not applicable
//             : -2, need skip, unique_index not accumulate
//             :  0, success
struct result_t{
    int return_code     {-1};
    int gks             {0};  // this is to store the gks value after benchmarked.
    int grid_size       {0};
    float duration_ms   {FLT_MAX};
    float gflops        {0};
    float efficiency    {0};
    std::string kernel_name;
    dumpheader_t dumpheader;
    std::vector<dispatchinfo_t> dumpdata;
} ;