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

//static std::pair<int,int> macro_tiles[] = { {256,128}, {128,256}, {256,64}, {64,256}, {256,32}, {32,256}, {128,128}, {128,64}, {64,128}, {128,32},{32,128},{256,16},{16,256}, 
//	                                    {64,64}, {64,32}, {32,64}, {128,16}, {16,128}, {32,32}, {64,16}, {16,64}, {64,8}, {8,64}, {32,16}, {16,32}, {64,4}, {4,64} }; 
//static std::pair<int,int> macro_tiles[] = { {256,128}, {128,256}, {256,64}, {64,256}, {256,32}, {32,256}, {128,128}, {128,64}, {64,128}, {128,32},{32,128}, 
//	                                    {64,64}, {64,32}, {32,64}, {32,32}, {64,16}, {16,64}, {64,8}, {8,64}, {32,16}, {16,32}, {64,4}, {4,64} }; 
	                                    
static std::pair<int,int> macro_tiles[] = { {128,256}, {256,128}, {64,256}, {256,64}, {32,256}, {256,32}, {128,128}, {64,128}, {128,64}, {32,128},{128,32}, 
	                                    {64,64}, {32,64}, {64,32}, {32,32}, {16,64}, {64,16}, {8,64}, {64,8}, {16,32}, {32,16}, {4,64}, {64,4} }; 

#define NUM_MACRO_TILES (sizeof(macro_tiles)/sizeof(macro_tiles[0]))

static std::vector<igemm_gtc_tunable_t> ordered_configs; 

struct sorterClass 
{
  bool operator()(igemm_gtc_tunable_t cfg1, igemm_gtc_tunable_t cfg2) 
  { 
     if ( (cfg1.gemm_k_per_block == 16 && cfg2.gemm_k_per_block == 8) || (cfg1.gemm_k_per_block == 8 && cfg2.gemm_k_per_block == 16) || cfg1.gemm_k_per_block == cfg2.gemm_k_per_block ) {
     //if ( cfg1.gemm_k_per_block == cfg2.gemm_k_per_block ) {
           if ( cfg1.tensor_a_cluster_lengths[3] * cfg1.tensor_b_cluster_lengths[3] > cfg2.tensor_a_cluster_lengths[3] * cfg2.tensor_b_cluster_lengths[3] )
                return(true); 
           
           if ( cfg1.tensor_a_cluster_lengths[3] * cfg1.tensor_b_cluster_lengths[3] < cfg2.tensor_a_cluster_lengths[3] * cfg2.tensor_b_cluster_lengths[3] )
                return(false); 
     }; 

     if ( (cfg1.gemm_k_per_block == 8 && cfg2.gemm_k_per_block == 4) || (cfg1.gemm_k_per_block == 4 && cfg2.gemm_k_per_block == 8) ) {
           if ( cfg1.tensor_a_cluster_lengths[3] * cfg1.tensor_b_cluster_lengths[3] > cfg2.tensor_a_cluster_lengths[3] * cfg2.tensor_b_cluster_lengths[3] )
                return(true);

           if ( cfg1.tensor_a_cluster_lengths[3] * cfg1.tensor_b_cluster_lengths[3] < cfg2.tensor_a_cluster_lengths[3] * cfg2.tensor_b_cluster_lengths[3] )
                return(false);
     };

     if ( cfg1.gemm_k_per_block > cfg2.gemm_k_per_block )
	  return(true); 

     if (cfg1.gemm_k_per_block < cfg2.gemm_k_per_block ) 
	  return(false); 

     if ( cfg1.nxb > cfg2.nxb ) 
	  return(true); 

     if ( cfg1.nxb < cfg2.nxb ) 
	  return(false); 

     // This is needed to ensure tunable with nxe==0 is selected for x=y=1
     // Check the tunable_is_valid() which is not touched by me 
     if ( cfg1.nxe <= cfg2.nxe )
	  return(true); 

     return(false); 
  };
} sorterObj; 

const char *arch="\'gfx908\'"; 
const char *code_object="\'cov3\'"; 
const char *mode = "\'flat\'"; 

static void output_configurations(std::vector<igemm_gtc_tunable_t> &configs, const char *strDirection, const char *strPrecision, std::ostream &myout)
{
    
    myout << "[codegen]" << std::endl; 
    myout << "arch = " << arch << std::endl; 
    myout << "code_object = " << code_object << std::endl; 
    myout << "mode = " << mode << std::endl; 

    myout << std::endl; 

    for (auto ite=configs.begin(); ite != configs.end(); ite++) {
	 myout << std::dec; 
         myout << std::endl;
         myout << "#--------------------------- " << ite->gemm_m_per_block << "x" << ite->gemm_n_per_block << std::endl;
	 myout << "[igemm_fwd_gtc]" << std::endl; 
         myout << "gemm_m_per_block         = " << ite->gemm_m_per_block << std::endl; 
         myout << "gemm_n_per_block         = " << ite->gemm_n_per_block << std::endl; 
         myout << "gemm_k_per_block         = " << ite->gemm_k_per_block << std::endl; 
         myout << "wave_tile_m              = " << ite->wave_tile_m << std::endl; 
         myout << "wave_step_m              = " << ite->wave_step_m << std::endl;
	 myout << "wave_repeat_m            = " << ite->wave_repeat_m << std::endl; 
         myout << "wave_tile_n              = " << ite->wave_tile_n << std::endl; 
         myout << "wave_step_n              = " << ite->wave_step_n << std::endl; 
	 myout << "wave_repeat_n            = " << ite->wave_repeat_n << std::endl; 

         myout << "tensor_a_thread_lengths  = [" << ite->tensor_a_thread_lengths[0] <<  ", " << ite->tensor_a_thread_lengths[1] << ", "; 
	 myout << ite->tensor_a_thread_lengths[2] << ", " << ite->tensor_a_thread_lengths[3]   << "]" << "       # C0xC1ExK0xK1" << std::endl; 
		 
         myout << "tensor_a_cluster_lengths = [" << ite->tensor_a_cluster_lengths[0] << ", " << ite->tensor_a_cluster_lengths[1] << ", "; 
	 myout << ite->tensor_a_cluster_lengths[2] << ", " << ite->tensor_a_cluster_lengths[3] << "]" << "       # C0xC1ExK0xK1" << std::endl; 
		 
         myout << "tensor_b_thread_lengths  = [" << ite->tensor_b_thread_lengths[0] <<  ", " << ite->tensor_b_thread_lengths[1] << ", "; 
	 myout << ite->tensor_b_thread_lengths[2] << ", " << ite->tensor_b_thread_lengths[3]   << "]" << "       # C0xC1ExN0xN1B" << std::endl; 
		 
         myout << "tensor_b_cluster_lengths = [" << ite->tensor_b_cluster_lengths[0] << ", " << ite->tensor_b_cluster_lengths[1] << ", "; 
	 myout << ite->tensor_b_cluster_lengths[2] << ", " << ite->tensor_b_cluster_lengths[3] << "]" << "       # C0xC1ExN0xN1B" << std::endl;

         myout << "direction                = " << strDirection << std::endl; 
	 myout << "precision                = " << strPrecision << std::endl; 

	 myout << "nxb                      = " << ite->nxb << std::endl; 
	 myout << "nxe                      = " << ite->nxe << std::endl; 
    };
}; 

int main(int argc, char **argv) 
{
    if ( argc != 3 ) {
         fprintf(stdout, "Usage: %s, <src configuration file> <dst configuration file>\n", argv[0]);
         return(-1); 
    }; 

    char *config_file = argv[1]; 

    config_parser_t config_parser(config_file);
    auto content = config_parser.parse();
    // content.dump();
   
    std::ofstream ofs (argv[2], std::ofstream::out);

    auto tunables = igemm_gtc_tunable_from_config(content);
    if (tunables.size() == 0){
        fprintf(stdout, "no tunable specified, may not work\n");
        return 0;
    }
    fprintf(stdout, "tunables:%d\n", (int)tunables.size());

    std::map< std::pair<int,int>, std::vector<igemm_gtc_tunable_t> > indexed_configs; 
    std::map< std::pair<int,int>, std::vector<igemm_gtc_tunable_t> >::iterator it;

    int count=0; 
    for (int i = 0; i < tunables.size(); i++) {
         igemm_gtc_tunable_t *tunable = &tunables[i];

         auto mt = std::make_pair((int)tunable->gemm_m_per_block, (int)tunable->gemm_n_per_block); 

	 it = indexed_configs.find(mt); 

         if ( it == indexed_configs.end() ) {
              std::vector<igemm_gtc_tunable_t> tmpVector; 	

              indexed_configs.insert( std::make_pair(mt, tmpVector) ); 
              it = indexed_configs.find(mt); 
         }

         assert(it != indexed_configs.end());  

         count++; 
         it->second.push_back(*tunable); 
    }

    fprintf(stdout, "%d configurations checked\n", count); 

    for (int i=0; i < NUM_MACRO_TILES; i++) {
         auto mt = macro_tiles[i]; 
        
	 it = indexed_configs.find(mt); 

         if ( it != indexed_configs.end() ) {
              fprintf(stdout, "Macro-tile [%d,%d], number of configurations %d\n", it->first.first, it->first.second, (int)it->second.size()); 	
    	      std::sort(it->second.begin(), it->second.end(), sorterObj); 	

              for (auto it_tunable = it->second.begin(); it_tunable != it->second.end(); it_tunable++) 
                   ordered_configs.push_back(*it_tunable); 
         }; 
    }; 

    fprintf(stdout, "\nSize of the orderred configs array %d\n", (int)ordered_configs.size()); 

    const char *strDirection = "\'fwd\'";
    const char *strPrecision = "\'fp32\'"; 
    output_configurations(ordered_configs, strDirection, strPrecision, ofs); 
};
