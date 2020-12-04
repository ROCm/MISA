/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all
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
#include "config_parser.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>
#include <algorithm>
#include <map> 
#include <memory>
#include <fstream>
#include <iostream>

#include "igemm_gtc_base.h"

typedef struct {
    int macro_tile_m;
    int macro_tile_n;
    int wave_tile_m;
    int wave_tile_n;
    int wave_tile_k;
    int waves;
    int wave_repeat_m;
    int wave_repeat_n;
    int wave_step_m;
    int wave_step_n;
} xdlops_mapping_t; 

static xdlops_mapping_t xdlops_mappings[] = {
        { 256, 128,  64,  32,  4, 4,  2,  2,  1,  1,  },
        { 256, 128,  32,  32,  8, 4,  2,  2,  2,  1,  },
        { 128, 256,  32,  64,  4, 4,  2,  2,  1,  1,  },
        { 128, 256,  32,  32,  8, 4,  2,  2,  1,  2,  },
        { 256, 64 ,  64,  16,  4, 4,  2,  2,  1,  1,  },
        { 64 , 256,  16,  64,  4, 4,  2,  2,  1,  1,  },
        { 64 , 256,  32,  64,  4, 4,  1,  1,  1,  2,  },
      /*  { 64 , 256,  32,  32,  8, 4,  1,  2,  1,  2,  }, */
        { 256, 32 ,  64,  4 ,  4, 4,  2,  2,  1,  2,  },
        { 32 , 256,  4 ,  64,  4, 4,  2,  2,  2,  1,  },
        { 256, 16 ,  64,  4 ,  4, 4,  2,  2,  1,  1,  },
        { 16 , 256,  4 ,  64,  4, 4,  2,  2,  1,  1,  },
        { 128, 128,  32,  32,  4, 4,  2,  2,  1,  1,  },
        { 128, 128,  32,  32,  8, 4,  2,  2,  1,  1,  },
        { 128, 128,  16,  16, 16, 4,  2,  2,  2,  2,  },
        { 128,  64,  16,  16, 16, 4,  2,  2,  2,  1,  },
        { 128, 128,  32,  64,  4, 4,  1,  1,  2,  1,  },
        { 128, 64 ,  32,  8 ,  4, 4,  2,  2,  1,  2,  },
        { 64 , 128,  8 ,  32,  4, 4,  2,  2,  2,  1,  },
        { 64 , 128,  32,  64,  4, 4,  1,  1,  1,  1,  },
        { 64 , 128,  64,  32,  4, 4,  1,  1,  1,  1,  },
        { 64 , 128,  32,  32,  8, 4,  1,  1,  1,  2,  },
        { 128, 32 ,  32,  8 ,  4, 4,  2,  2,  1,  1,  },
        { 32 , 128,  8 ,  32,  4, 4,  2,  2,  1,  1,  },
        { 32 , 128,  16,  64,  4, 4,  1,  1,  1,  1,  },
        { 64 , 64 ,  16,  16,  4, 4,  2,  2,  1,  1,  },
        { 64 , 64 ,  16,  16, 16, 4,  2,  2,  1,  1,  },
        { 64 , 64 ,  16,  16, 16, 4,  1,  1,  2,  2,  },
        { 128, 16 ,  64,  16,  4, 2,  1,  1,  1,  1,  },
        { 16 , 128,  16,  64,  4, 2,  1,  1,  1,  1,  },
        { 64 , 32 ,  32,  8 ,  4, 4,  1,  1,  1,  2,  },
        { 32 , 64 ,  8 ,  32,  4, 4,  1,  1,  2,  1,  },
        { 32 , 32 ,  16,  16,  4, 4,  1,  1,  1,  1,  },
        { 32 , 32 ,  16,  16, 16, 4,  1,  1,  1,  1,  },
        { 64 , 16 ,  64,  4 ,  4, 4,  1,  1,  1,  1,  },
        { 16 , 64 ,  4 ,  64,  4, 4,  1,  1,  1,  1,  },
        { 64 , 16 ,  64,  4 ,  4, 2,  1,  1,  1,  2,  },
        { 16 , 64 ,  4 ,  64,  4, 2,  1,  1,  2,  1,  },
        { 64 , 8  ,  64,  4 ,  4, 2,  1,  1,  1,  1,  },
        { 8  , 64 ,  4 ,  64,  4, 2,  1,  1,  1,  1,  },
        { 32 , 16 ,  32,  8 ,  4, 2,  1,  1,  1,  1,  },
        { 16 , 32 ,  8 ,  32,  4, 2,  1,  1,  1,  1,  },
        { 32 , 16 ,  32,  8 ,  4, 1,  1,  1,  1,  2,  },
        { 16 , 32 ,  8 ,  32,  4, 1,  1,  1,  2,  1,  },
        { 64 , 4 ,  64,  4 ,   4, 1,  1,  1,  1,  1,  },
        { 4  , 64,  4 ,  64,   4, 1,  1,  1,  1,  1,  },
        { 16 , 16,  16,  16,   4, 1,  1,  1,  1,  1,  },
}; 

#define NUM_XDLOPS_MAPPING (sizeof(xdlops_mappings)/sizeof(xdlops_mapping_t))

static const int waveSize = 64; 

static inline void output_single_config(const igemm_gtc_tunable_t & cfg, const char *strDirection, const char *strPrecision, std::ostream &myout)
{
         myout << "#--------------------------- " << cfg.gemm_m_per_block << "x" << cfg.gemm_n_per_block << std::endl;
         myout << "[igemm_fwd_gtc]" << std::endl;
         myout << "gemm_m_per_block         = " << cfg.gemm_m_per_block << std::endl;
         myout << "gemm_n_per_block         = " << cfg.gemm_n_per_block << std::endl;
         myout << "gemm_k_per_block         = " << cfg.gemm_k_per_block << std::endl;
         myout << "wave_tile_m              = " << cfg.wave_tile_m << std::endl;
         myout << "wave_step_m              = " << cfg.wave_step_m << std::endl;
         myout << "wave_repeat_m            = " << cfg.wave_repeat_m << std::endl;
         myout << "wave_tile_n              = " << cfg.wave_tile_n << std::endl;
         myout << "wave_step_n              = " << cfg.wave_step_n << std::endl;
         myout << "wave_repeat_n            = " << cfg.wave_repeat_n << std::endl;
         myout << "wave_tile_k              = " << cfg.wave_tile_k << std::endl;

         myout << "tensor_a_thread_lengths  = [" << cfg.tensor_a_thread_lengths[0] <<  ", " << cfg.tensor_a_thread_lengths[1] << ", ";
         myout << cfg.tensor_a_thread_lengths[2] << ", " << cfg.tensor_a_thread_lengths[3]   << "]" << "       # C0xC1ExK0xK1" << std::endl;

         myout << "tensor_a_cluster_lengths = [" << cfg.tensor_a_cluster_lengths[0] << ", " << cfg.tensor_a_cluster_lengths[1] << ", ";
         myout << cfg.tensor_a_cluster_lengths[2] << ", " << cfg.tensor_a_cluster_lengths[3] << "]" << "       # C0xC1ExK0xK1" << std::endl;

         myout << "tensor_b_thread_lengths  = [" << cfg.tensor_b_thread_lengths[0] <<  ", " << cfg.tensor_b_thread_lengths[1] << ", ";
         myout << cfg.tensor_b_thread_lengths[2] << ", " << cfg.tensor_b_thread_lengths[3]   << "]" << "       # C0xC1ExN0xN1B" << std::endl;

         myout << "tensor_b_cluster_lengths = [" << cfg.tensor_b_cluster_lengths[0] << ", " << cfg.tensor_b_cluster_lengths[1] << ", ";
         myout << cfg.tensor_b_cluster_lengths[2] << ", " << cfg.tensor_b_cluster_lengths[3] << "]" << "       # C0xC1ExN0xN1B" << std::endl;

         myout << "direction                = " << strDirection << std::endl;
         myout << "precision                = " << strPrecision << std::endl;

         myout << "nxb                      = " << cfg.nxb << std::endl;
         myout << "nxe                      = " << cfg.nxe << std::endl;
}; 

static void output_configurations(std::vector<igemm_gtc_tunable_t> &configs, const char *strDirection, const char *strPrecision, std::ostream &myout)
{
    static const char *arch="\'gfx908\'"; 
    static const char *code_object="\'cov3\'"; 
    static const char *mode = "\'flat\'"; 
    
    myout << "[codegen]" << std::endl; 
    myout << "arch = " << arch << std::endl; 
    myout << "code_object = " << code_object << std::endl; 
    myout << "mode = " << mode << std::endl; 

    myout << std::endl; 

    for (const auto& cfg : configs) {
	 myout << std::dec; 
         myout << std::endl;
         output_single_config(cfg, strDirection, strPrecision, myout); 
    };
}; 


static int getMaximumSlice_a_c1e(int gemm_k_per_block, int blockSize, int macro_tile_m)
{
    int a_slice_size=8; 

    for (; a_slice_size > 1; a_slice_size /= 2) 
        if ( blockSize / (gemm_k_per_block/a_slice_size) < macro_tile_m ) 
             break; 	

    return(a_slice_size); 
}; 

static int getMaximumCluster_b_n1b(int gemm_k_per_block, int blockSize, int macro_tile_n)
{
    int b_cluster_size = std::min(blockSize, macro_tile_n);

    for (; b_cluster_size > 1; b_cluster_size /= 2) 
        if ( blockSize / b_cluster_size <= gemm_k_per_block )
	     break; 
 
    return(b_cluster_size); 
}; 

static std::vector<igemm_gtc_tunable_t> configs;

int main(int argc, char **argv) 
{
    if ( argc != 2 ) {
         fprintf(stdout, "Usage: %s, <output configuration file> \n", argv[0]);
         return(-1); 
    }; 

    const char *config_file = argv[1]; 

    std::ofstream ofs(argv[1], std::ofstream::out);

    for (int i=0; i < NUM_XDLOPS_MAPPING; i++) {
         auto xm = xdlops_mappings[i]; 

         igemm_gtc_tunable_t cfg; 

         cfg.gemm_m_per_block = xm.macro_tile_m; 
         cfg.gemm_n_per_block = xm.macro_tile_n; 
         cfg.wave_tile_m = xm.wave_tile_m; 
	 cfg.wave_tile_n = xm.wave_tile_n; 
         cfg.wave_tile_k = xm.wave_tile_k; 
         cfg.wave_repeat_m = xm.wave_repeat_m; 
	 cfg.wave_repeat_n = xm.wave_repeat_n; 
	 cfg.wave_step_m = xm.wave_step_m; 
	 cfg.wave_step_n = xm.wave_step_n; 

         cfg.tensor_a_thread_lengths.resize(4); 
         cfg.tensor_a_cluster_lengths.resize(4); 
         cfg.tensor_b_thread_lengths.resize(4); 
         cfg.tensor_b_cluster_lengths.resize(4); 

         int blockSize = waveSize * xm.waves; 

         for (int nxe=0; nxe < 2; nxe += 1)  {
              cfg.nxe = nxe; 	
	      for (int nxb=1; nxb < 17; nxb *= 4) {
                   if ( cfg.gemm_n_per_block % nxb != 0 ) 
			continue;  

                   cfg.nxb = nxb;

                   // consider the gemm_k_per_block sizes to be 2x, 4x, 8x that of the k_per_inst of the specific xlops instruction
                   for (int k=1; k < 3; k++) {
                        cfg.gemm_k_per_block = xm.wave_tile_k << k;             

                        if ( cfg.gemm_k_per_block / 8 > blockSize )  // this should not occurr easily
		             continue;  

                        if ( blockSize / (cfg.gemm_k_per_block/1) > cfg.gemm_m_per_block ) // this could occurr easily for small value of gemm_m_per_block 
			     continue; 	

                        if ( blockSize / std::min(blockSize, cfg.gemm_n_per_block) > cfg.gemm_k_per_block )
			     continue; 

                        int slice_a_c1e = getMaximumSlice_a_c1e(cfg.gemm_k_per_block, blockSize, cfg.gemm_m_per_block); 

	                cfg.tensor_a_thread_lengths[0] = 1; 
	                cfg.tensor_a_thread_lengths[1] = slice_a_c1e; 
                        cfg.tensor_a_cluster_lengths[0] = 1; 
	                cfg.tensor_a_cluster_lengths[1] = cfg.gemm_k_per_block / slice_a_c1e; 
                        cfg.tensor_a_cluster_lengths[2] = 1; 
	                cfg.tensor_a_cluster_lengths[3] = blockSize / cfg.tensor_a_cluster_lengths[1]; 
	                cfg.tensor_a_thread_lengths[2] = cfg.gemm_m_per_block / cfg.tensor_a_cluster_lengths[3]; 
                        cfg.tensor_a_thread_lengths[3] = 1; 

                        int cluster_b_n1b = getMaximumCluster_b_n1b(cfg.gemm_k_per_block, blockSize, cfg.gemm_n_per_block); 

		        cfg.tensor_b_cluster_lengths[3] = cluster_b_n1b; 
		        cfg.tensor_b_cluster_lengths[2] = 1;
		        cfg.tensor_b_cluster_lengths[1] = blockSize / cluster_b_n1b; 
		        cfg.tensor_b_cluster_lengths[0] = 1; 
			cfg.tensor_b_thread_lengths[0] = 1; 
			cfg.tensor_b_thread_lengths[1] = cfg.gemm_k_per_block / cfg.tensor_b_cluster_lengths[1];
			cfg.tensor_b_thread_lengths[2] = cfg.gemm_n_per_block / cfg.tensor_b_cluster_lengths[3];  
			cfg.tensor_b_thread_lengths[3] = 1; 

                        configs.push_back(cfg); 

                        // we need a config which has tensor_b_thread_lengths[1] = 1 to support the cases where either x != 1 or y != 1
                        if ( cfg.tensor_b_cluster_lengths[1] != cfg.gemm_k_per_block && blockSize / cfg.gemm_k_per_block <= cfg.gemm_n_per_block ) {
                             cfg.tensor_b_thread_lengths[0] = 1; 
			     cfg.tensor_b_thread_lengths[1] = 1; 
			     cfg.tensor_b_cluster_lengths[0] = 1; 
                             cfg.tensor_b_cluster_lengths[1] = cfg.gemm_k_per_block; 
                             cfg.tensor_b_cluster_lengths[2] = 1; 
                             cfg.tensor_b_cluster_lengths[3] = blockSize / cfg.tensor_b_cluster_lengths[1]; 
			     cfg.tensor_b_thread_lengths[2] = cfg.gemm_n_per_block / cfg.tensor_b_cluster_lengths[3]; 
			     cfg.tensor_b_thread_lengths[3] = 1; 

                             // to satisfy unmerge_sub_n % nb_n0 == 0 
                             if ( (cfg.gemm_n_per_block / cfg.nxb ) % cfg.tensor_b_thread_lengths[2] == 0 ) 
			           configs.push_back(cfg); 
			}; 
                   }; 
	     }; 
         };	
    };  

    const char *strDirection = "\"fwd\"";
    const char *strPrecision = "\'fp16\'";

    output_configurations(configs, strDirection, strPrecision, ofs);
};
