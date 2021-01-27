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

static void output_h_file(std::vector<igemm_gtc_tunable_t> &configs, const char *strDirection, const char *strPrecision, std::ostream &myout)
{
    static const char *dataItemName = "TunableImplicitGemmGTCDynamic_t"; 
    static const char *comma = ",  "; 
    static const char *ident = "    "; 

    myout << "static inline std::vector<" << dataItemName << ">& " << std::endl; 

    if ( std::string(strDirection) == "fwd" )    
         myout << "GetImplicitGemmGtcDynamicFwdXdlopsTunablesList()" << std::endl; 
    else
         myout << "GetImplicitGemmGtcDynamicBwdXdlopsTunablesList()" << std::endl; 

    myout << "{" << std::endl; 

    myout << ident << "// list all the dynamic igemm conv-fwd kernels" << std::endl; 
    myout << ident << "// clang-format off" << std::endl; 

    myout << ident << "static std::vector<TunableImplicitGemmGTCDynamic_t> kernel_param_list {" << std::endl; 

    for (const auto& cfg : configs) {
         myout << ident << ident << "{ "; 

         myout << "\"" << strDirection << "\"" << comma; 
	 myout << "\"" << strPrecision << "\"" << comma;

	 myout << cfg.nxb << comma;
	 myout << cfg.nxe << comma;

         myout << cfg.gemm_m_per_block << comma << cfg.gemm_n_per_block << comma << cfg.gemm_k_per_block << comma; 

         myout << cfg.wave_tile_m << comma << cfg.wave_tile_n << comma << cfg.wave_tile_k << comma; 	 

         myout << cfg.wave_step_m << comma << cfg.wave_step_n << comma; 	 

         myout << cfg.wave_repeat_m << comma << cfg.wave_repeat_n << comma; 	 

         myout << '{' << cfg.tensor_a_thread_lengths[0] << comma << cfg.tensor_a_thread_lengths[1] << comma;
	 myout << cfg.tensor_a_thread_lengths[2] << comma << cfg.tensor_a_thread_lengths[3] << '}' << comma;  
         myout << '{' << cfg.tensor_a_cluster_lengths[0] << comma << cfg.tensor_a_cluster_lengths[1] << comma;
	 myout << cfg.tensor_a_cluster_lengths[2] << comma << cfg.tensor_a_cluster_lengths[3] << '}' << comma; 

         myout << '{' << cfg.tensor_b_thread_lengths[0] << comma << cfg.tensor_b_thread_lengths[1] << comma;
	 myout << cfg.tensor_b_thread_lengths[2] << comma << cfg.tensor_b_thread_lengths[3] << '}' << comma; 
         myout << '{' << cfg.tensor_b_cluster_lengths[0] << comma << cfg.tensor_b_cluster_lengths[1] << comma;
	 myout << cfg.tensor_b_cluster_lengths[2] << comma << cfg.tensor_b_cluster_lengths[3] << '}' << comma; 

         myout << "0"; 

         myout << " }" << comma << std::endl; 	 
    };  

    myout << ident << '}' << ';' << std::endl; 
    myout << std::endl;
    myout << ident << "return kernel_param_list;" << std::endl; 
    myout << '}' << std::endl; 
}; 

int main(int argc, char **argv) 
{
    if ( argc != 3 ) {
         fprintf(stdout, "Usage: %s, <configuration file> <C++ vector of Tuables> \n", argv[0]);
         return(-1); 
    }; 

    const char *config_file = argv[1]; 
    const char *tunables_h_file = argv[2];  

    config_parser_t config_parser(config_file);
    auto content = config_parser.parse();
    // content.dump();
   
    std::ofstream ofs(argv[2], std::ofstream::out);

    auto tunables = igemm_gtc_tunable_from_config(content);
    if (tunables.size() == 0){
        fprintf(stdout, "no tunable specified, may not work\n");
        return 0;
    }
    fprintf(stdout, "tunables:%d\n", (int)tunables.size());

    const char *strDirection = "bwd";

#ifdef USE_PRECISION_FP16    
    const char *strPrecision = "fp16"; 
#else    
    const char *strPrecision = "fp32"; 
#endif    

    output_h_file(tunables, strDirection, strPrecision, ofs); 
};

