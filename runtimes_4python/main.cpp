#include <iostream>
#include "log.hpp"
#include "runtime.hpp"
#include "op_kernel_args.hpp"


int main(){
    LOG.SetSeverity(4);
    try{
        Runtime rt;
        rt.init_py();
        std::string st;
        
        std::cout << "No aloc";
        
        auto in1 = rt.create_buf(memtag{"some in 1", 1LL*1024*1024*1024, usage::IN, true});
        auto in2 = rt.create_buf(memtag{"some in 2", 1LL*1024*1024*2048, usage::IN, true});
        auto out1 = rt.create_buf(memtag{"some out 1", 1024*1, usage::OUT, true});
        std::cout << "send some key:";
        std::cin >> st;
        rt.memset_buf(in1, 1, 256 * 4);

        std::string path = "/home/kamil/igemm/iGEMMgen/runtimes_4python/build/add_1.o";
        std::string kernel_name = "add_1_gfx9";

        kernel kernel;
        rt.load_kernel_from_binary(&kernel, path, kernel_name);
        std::cout << "kernel:" << kernel.name << " loaded \n";
        
        dispatch_params params;
        
        params.wg_size = {128, 1, 1};
        params.grid_size = {2, 1, 1};
        params.dynamic_lds = 0;
        
        kernel_args_constructor kac;

        kac.push_back_ptr(in1.ptr);
        kac.push_back_ptr(out1.ptr);

        params.kernarg_size = kac.get_size();
        params.kernarg = kac.create_array();

        //struct
        //{
        //    uint64_t in_addr;
        //    uint64_t out_addr;
        //} add_params;
        //add_params.in_addr = (uint64_t) (in1.ptr);
        //add_params.out_addr = (uint64_t) (out1.ptr);
        //params.kernarg_size = 16;
        //params.kernarg = &add_params;
        
        rt.short_kernel_run(&kernel, &params);

        std::vector<int> out(257);
        bufview out_v(out1);
        rt.copy_mem(out.data(), &out_v, 256*4);

        for(int i =0; i < 200; i++)
            std::cout << "i:" << i <<"=" << out[i]<<'\n';

        rt.delete_buf(in1);
        rt.delete_buf(in2);

    }
    catch(std::runtime_error& e)
    {
        std::cout << e.what() << "\n";
    }
    return 0;
}