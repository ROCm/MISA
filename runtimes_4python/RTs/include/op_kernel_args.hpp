#pragma once

#include <type_traits>
#include <cstdint>
#include <half.hpp>
#include <vector>

#include <boost/container/small_vector.hpp>

struct OpKernelArg
{

    OpKernelArg(char val, size_t sz) : buffer(sz) { std::fill(buffer.begin(), buffer.end(), val); }

    template <typename T>
    OpKernelArg(T arg) : buffer(sizeof(T))
    {
        static_assert(std::is_trivial<T>{} || std::is_same<T, half_float::half>{},
                      "Only for trivial types");
        *(reinterpret_cast<T*>(buffer.data())) = arg;
    }

    template <typename T>
    OpKernelArg(T* arg) // NOLINT
        : buffer(sizeof(T*))
    {
        *(reinterpret_cast<T**>(buffer.data())) = arg;
        is_ptr                                  = true;
    }

    std::size_t size() const { return buffer.size(); };
    boost::container::small_vector<char, 8> buffer;
    bool is_ptr = false;
};

class kernel_args_constructor
{
    std::vector<OpKernelArg> any_args;
    bool is_changed = true;

    size_t last_arg_size = 0;
    bool is_changed_size = true;

    size_t get_any_args_size() const{
        auto sz_left       = any_args[0].size();
        for(unsigned long idx = 1; idx < any_args.size(); idx++)
        {
            auto& any_arg              = any_args[idx];
            unsigned long alignment    = any_arg.size();
            unsigned long padding      = (alignment - (sz_left % alignment)) % alignment;
            unsigned long second_index = sz_left + padding;
            sz_left = second_index + alignment;
        }
        return sz_left;
    }

    std::vector<char> args_vector;

public:
    kernel_args_constructor(){};

    void * create_array()
    {
        if(!is_changed)
            return args_vector.data();

        is_changed = false;

        get_size();
        args_vector.resize(last_arg_size);

        char * hip_args = args_vector.data();

        auto sz_left       = any_args[0].size();

        memcpy(hip_args, &(any_args[0].buffer[0]), any_args[0].size());
        //        copy_arg(any_args[0], hip_args, 0);

        for(unsigned long idx = 1; idx < any_args.size(); idx++)
        {
            auto& any_arg              = any_args[idx];
            unsigned long alignment    = any_arg.size();
            unsigned long padding      = (alignment - (sz_left % alignment)) % alignment;
            unsigned long second_index = sz_left + padding;
            memcpy(hip_args + second_index, &(any_arg.buffer[0]), any_arg.size());
            sz_left = second_index + alignment;
        }
        return hip_args;
    }

    size_t get_size()
    {
        if(is_changed_size)
            last_arg_size = get_any_args_size();
        is_changed_size = false;
        return last_arg_size;
    }

    void push_back_ptr(void * ptr){ any_args.push_back(ptr); is_changed = true; is_changed_size = true;}
    void push_back_int_32(int val){ any_args.push_back(val); is_changed = true; is_changed_size = true;}
    void push_back_sizet_64(std::size_t val) { any_args.push_back(val); is_changed = true; is_changed_size = true;}
    void push_back_float_32(float val) { any_args.push_back(val); is_changed = true; is_changed_size = true;}
};
