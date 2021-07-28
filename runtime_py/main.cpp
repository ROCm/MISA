#include <iostream>
#include "log.hpp"
#include "runtime.hpp"


int main(){

    try{
        Runtime rt;
        rt.init_py();
        std::string st;
        
        std::cout << "No aloc";
        auto out1 = rt.create_buf(memtag{"some out 1", 1024*1, usage::OUT, true});
        std::cin >> st;
        auto in1 = rt.create_buf(memtag{"some in 1", 1LL*1024*1024*1024, usage::IN, true});
        std::cin >> st;
        auto in2 = rt.create_buf(memtag{"some in 2", 1LL*1024*1024*2048, usage::IN, true});
        std::cin >> st;
        
        
        

        std::cin >> st;
        rt.delete_buf(in1);
        std::cin >> st;
        rt.delete_buf(in2);
        std::cin >> st;
    }
    catch(std::runtime_error& e)
    {
        std::cout << e.what() << "\n";
    }
    return 0;
}